import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JudgeService } from './lib/judge-service.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const workspaceRoot = path.resolve(__dirname, '..');
const publicDir = path.join(workspaceRoot, 'public');

const judgeService = new JudgeService({ workspaceRoot });

function json(response, statusCode, payload) {
  response.writeHead(statusCode, { 'Content-Type': 'application/json; charset=utf-8' });
  response.end(JSON.stringify(payload));
}

function notFound(response) {
  json(response, 404, { error: 'Not found' });
}

async function readJsonBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }
  if (!chunks.length) {
    return {};
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  return JSON.parse(raw);
}

/**
 * Parse a multipart/form-data request.
 * Returns { fields: Record<string, string>, files: Record<string, { name, buffer }> }.
 * Operates on Buffers throughout to avoid V8 string-length limits on large uploads.
 */
async function readMultipartBody(request) {
  const contentType = request.headers['content-type'] || '';
  const boundaryMatch = contentType.match(/boundary=([^\s;]+)/);
  if (!boundaryMatch) {
    throw new Error('Missing multipart boundary');
  }
  const boundary = boundaryMatch[1];
  const delimiter = Buffer.from(`--${boundary}`);

  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }
  const body = Buffer.concat(chunks);

  const fields = {};
  const files = {};
  const CRLF = Buffer.from('\r\n');
  const CRLFCRLF = Buffer.from('\r\n\r\n');

  let offset = bufferIndexOf(body, delimiter, 0);
  if (offset === -1) {
    return { fields, file };
  }

  while (true) {
    // Move past the delimiter
    offset += delimiter.length;
    // Check for closing delimiter (--)
    if (body[offset] === 0x2d && body[offset + 1] === 0x2d) {
      break;
    }
    // Skip the CRLF after delimiter
    offset += CRLF.length;

    // Find end of headers
    const headersEnd = bufferIndexOf(body, CRLFCRLF, offset);
    if (headersEnd === -1) {
      break;
    }
    const headerBlock = body.subarray(offset, headersEnd).toString('utf8');
    const bodyStart = headersEnd + CRLFCRLF.length;

    // Find the next delimiter to know where the part body ends
    const nextDelim = bufferIndexOf(body, delimiter, bodyStart);
    if (nextDelim === -1) {
      break;
    }
    // Part body ends before \r\n--boundary
    const partBody = body.subarray(bodyStart, nextDelim - CRLF.length);

    // Parse Content-Disposition
    const nameMatch = headerBlock.match(/name="([^"]+)"/);
    const fileNameMatch = headerBlock.match(/filename="([^"]+)"/);

    if (fileNameMatch) {
      if (nameMatch) {
        files[nameMatch[1]] = { name: fileNameMatch[1], buffer: partBody };
      }
    } else if (nameMatch) {
      fields[nameMatch[1]] = partBody.toString('utf8');
    }

    offset = nextDelim;
  }

  return { fields, files };
}

function bufferIndexOf(buf, needle, start) {
  for (let i = start; i <= buf.length - needle.length; i++) {
    let found = true;
    for (let j = 0; j < needle.length; j++) {
      if (buf[i + j] !== needle[j]) {
        found = false;
        break;
      }
    }
    if (found) {
      return i;
    }
  }
  return -1;
}

function contentTypeFor(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  switch (ext) {
    case '.html': return 'text/html; charset=utf-8';
    case '.css': return 'text/css; charset=utf-8';
    case '.js': return 'application/javascript; charset=utf-8';
    case '.json': return 'application/json; charset=utf-8';
    case '.svg': return 'image/svg+xml';
    default: return 'application/octet-stream';
  }
}

async function serveStatic(request, response) {
  let requestPath = request.url;
  if (requestPath === '/') {
    requestPath = '/index.html';
  } else if (/^\/benchmarks\/[^/.]+\/?$/.test(requestPath)) {
    requestPath = '/benchmark.html';
  }
  const filePath = path.normalize(path.join(publicDir, requestPath));
  if (!filePath.startsWith(publicDir)) {
    notFound(response);
    return;
  }
  try {
    const info = await stat(filePath);
    if (!info.isFile()) {
      notFound(response);
      return;
    }
    const data = await readFile(filePath);
    response.writeHead(200, { 'Content-Type': contentTypeFor(filePath) });
    response.end(data);
  } catch {
    notFound(response);
  }
}

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://127.0.0.1');

    if (request.method === 'GET' && url.pathname === '/api/config') {
      await judgeService.init();
      json(response, 200, judgeService.getPublicConfig());
      return;
    }

    if (request.method === 'POST' && url.pathname === '/api/submissions') {
      const ct = request.headers['content-type'] || '';
      let body;
      if (ct.includes('multipart/form-data')) {
        const { fields, files } = await readMultipartBody(request);
        body = { ...fields };
        if (files.h5adFile) {
          body.fileName = files.h5adFile.name;
          body.fileBuffer = files.h5adFile.buffer;
        }
        if (files.datasetFile) {
          body.datasetFileName = files.datasetFile.name;
          body.datasetFileBuffer = files.datasetFile.buffer;
        }
      } else {
        body = await readJsonBody(request);
      }
      const job = await judgeService.submit(body);
      json(response, 202, job);
      return;
    }

    if (request.method === 'GET' && url.pathname.startsWith('/api/submissions/')) {
      const id = url.pathname.split('/').at(-1);
      const job = await judgeService.getJob(id);
      if (!job) {
        notFound(response);
        return;
      }
      json(response, 200, job);
      return;
    }

    if (request.method === 'GET') {
      await serveStatic(request, response);
      return;
    }

    notFound(response);
  } catch (error) {
    json(response, 500, {
      error: error instanceof Error ? error.message : 'Unknown server error',
    });
  }
});

const port = Number(process.env.PORT || 3000);
server.listen(port, () => {
  console.log(`Judge platform running at http://localhost:${port}`);
});
