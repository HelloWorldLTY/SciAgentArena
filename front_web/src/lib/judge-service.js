import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';

const MAX_INLINE_SOURCE_BYTES = 1_000_000;

function nowIso() {
  return new Date().toISOString();
}

function createId() {
  return crypto.randomUUID();
}

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

async function ensureDir(targetPath) {
  await mkdir(targetPath, { recursive: true });
}

async function loadJson(filePath) {
  const raw = await readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

function sanitizeFilename(name, fallback) {
  const safe = normalizeText(name).replace(/[^a-zA-Z0-9._-]+/g, '_');
  return safe || fallback;
}

function scoreAverage(summary) {
  const values = Object.values(summary ?? {}).filter((value) => Number.isFinite(value));
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

async function runCommand({ command, args, cwd, timeoutMs, env }) {
  return await new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let settled = false;

    const killTimer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill('SIGKILL');
      resolve({ ok: false, exitCode: null, stdout, stderr: `${stderr}\nExecution timed out after ${timeoutMs} ms.`.trim() });
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(killTimer);
      resolve({ ok: false, exitCode: null, stdout, stderr: `${stderr}\n${error.message}`.trim() });
    });

    child.on('close', (exitCode) => {
      if (settled) return;
      settled = true;
      clearTimeout(killTimer);
      resolve({ ok: exitCode === 0, exitCode, stdout, stderr });
    });
  });
}

export class JudgeService {
  constructor({ workspaceRoot }) {
    this.workspaceRoot = workspaceRoot;
    this.jobsDir = path.join(workspaceRoot, 'var', 'jobs');
    this.templatePath = path.join(workspaceRoot, 'templates', 'pipeline_evaluator.py');
    this.configPath = path.join(workspaceRoot, 'judge.config.json');
    this.jobs = new Map();
    this.queue = [];
    this.processing = false;
  }

  async init() {
    if (this.initialized) {
      return;
    }
    await ensureDir(this.jobsDir);
    this.config = await loadJson(this.configPath);
    this.initialized = true;
  }

  getPublicConfig() {
    if (!this.initialized) {
      throw new Error('JudgeService not initialized');
    }
    return {
      platformName: this.config.platformName,
      maxUploadSizeMb: this.config.maxUploadSizeMb,
      executionTimeoutSeconds: this.config.executionTimeoutSeconds,
      benchmarks: this.config.benchmarks,
      datasets: this.config.datasets.map(({ id, benchmark, label, datasetPath, markersPath, trajectoryPath, svgReferences, controlLabel, topK }) => ({
        id,
        benchmark,
        label,
        datasetPath,
        markersPath,
        trajectoryPath,
        svgReferences,
        controlLabel,
        topK,
      })),
      submissionModes: [
        { id: 'code', label: 'Python pipeline code' },
        { id: 'h5ad', label: 'Precomputed .h5ad output' },
      ],
    };
  }

  async submit(payload) {
    await this.init();
    const mode = normalizeText(payload.mode);
    const datasetId = normalizeText(payload.datasetId);
    const dataset = this.config.datasets.find((item) => item.id === datasetId);
    if (!dataset) {
      throw new Error('A valid dataset preset is required.');
    }
    if (!['code', 'h5ad'].includes(mode)) {
      throw new Error('Submission mode must be either code or h5ad.');
    }

    const jobId = createId();
    const jobDir = path.join(this.jobsDir, jobId);
    await ensureDir(jobDir);

    // Parse selected tasks – accepts a JSON array string or comma-separated ids
    let selectedTasks = null;
    if (payload.tasks) {
      try {
        selectedTasks = typeof payload.tasks === 'string' ? JSON.parse(payload.tasks) : payload.tasks;
      } catch {
        selectedTasks = String(payload.tasks).split(',').map((t) => t.trim()).filter(Boolean);
      }
      if (!Array.isArray(selectedTasks) || selectedTasks.length === 0) {
        selectedTasks = null; // treat empty as "all"
      }
    }

    const datasetConfig = {
      benchmark: dataset.benchmark,
      datasetLabel: dataset.label,
      datasetPath: normalizeText(payload.customDatasetPath) || dataset.datasetPath,
      markersPath: normalizeText(payload.customMarkersPath) || dataset.markersPath,
      trajectoryPath: normalizeText(payload.customTrajectoryPath) || dataset.trajectoryPath,
      svgReferences: dataset.svgReferences || {},
      hvgRange: dataset.hvgRange || [1900, 2100],
      controlLabel: dataset.controlLabel || 'ctrl',
      topK: dataset.topK || 20,
      tasks: selectedTasks,
    };

    const job = {
      id: jobId,
      status: 'queued',
      mode,
      datasetId: dataset.id,
      datasetLabel: dataset.label,
      createdAt: nowIso(),
      updatedAt: nowIso(),
      progress: { stage: 'queued', detail: 'Submission accepted.' },
      result: null,
      error: null,
      logs: [],
    };

    const maxBytes = this.config.maxUploadSizeMb * 1024 * 1024;

    if (Buffer.isBuffer(payload.datasetFileBuffer)) {
      if (payload.datasetFileBuffer.byteLength > maxBytes) {
        throw new Error(`Dataset upload exceeds ${this.config.maxUploadSizeMb} MB.`);
      }
      const datasetFileName = sanitizeFilename(payload.datasetFileName, 'dataset.h5ad');
      const uploadedDatasetPath = path.join(jobDir, datasetFileName);
      await writeFile(uploadedDatasetPath, payload.datasetFileBuffer);
      datasetConfig.datasetPath = uploadedDatasetPath;
    }

    if (mode === 'code') {
      const sourceCode = String(payload.sourceCode || '');
      if (!sourceCode.trim()) {
        throw new Error('Source code cannot be empty.');
      }
      if (Buffer.byteLength(sourceCode, 'utf8') > MAX_INLINE_SOURCE_BYTES) {
        throw new Error('Source code is too large for inline submission.');
      }
      await writeFile(path.join(jobDir, 'submission.py'), sourceCode, 'utf8');
    } else {
      if (!Buffer.isBuffer(payload.fileBuffer)) {
        throw new Error('Please upload a .h5ad prediction file.');
      }
      if (payload.fileBuffer.byteLength > maxBytes) {
        throw new Error(`Upload exceeds ${this.config.maxUploadSizeMb} MB.`);
      }
      const fileName = sanitizeFilename(payload.fileName, 'submission.h5ad');
      await writeFile(path.join(jobDir, fileName), payload.fileBuffer);
      datasetConfig.predictionFileName = fileName;
    }

    await writeFile(path.join(jobDir, 'dataset-config.json'), JSON.stringify(datasetConfig, null, 2), 'utf8');
    this.jobs.set(jobId, job);
    this.queue.push(jobId);
    this.processQueue().catch((error) => {
      console.error('Judge queue failure', error);
    });

    return job;
  }

  async getJob(id) {
    await this.init();
    if (this.jobs.has(id)) {
      return this.jobs.get(id);
    }
    try {
      const data = await loadJson(path.join(this.jobsDir, id, 'job.json'));
      this.jobs.set(id, data);
      return data;
    } catch {
      return null;
    }
  }

  async processQueue() {
    await this.init();
    if (this.processing) {
      return;
    }
    this.processing = true;

    while (this.queue.length) {
      const jobId = this.queue.shift();
      const job = this.jobs.get(jobId);
      if (!job) {
        continue;
      }
      try {
        await this.runJob(job);
      } catch (error) {
        job.status = 'failed';
        job.updatedAt = nowIso();
        job.error = error instanceof Error ? error.message : 'Unknown judge error';
        job.progress = { stage: 'failed', detail: job.error };
        await this.persistJob(job);
      }
    }

    this.processing = false;
  }

  async runJob(job) {
    const jobDir = path.join(this.jobsDir, job.id);
    const datasetConfig = await loadJson(path.join(jobDir, 'dataset-config.json'));
    const evaluatorPath = path.join(jobDir, 'pipeline_evaluator.py');
    const resultPath = path.join(jobDir, 'result.json');
    const metadataPath = path.join(jobDir, 'metadata.json');
    const pythonExecutable = process.env.PYTHON_EXECUTABLE || this.config.pythonExecutable || 'python';

    const metadata = {
      jobId: job.id,
      mode: job.mode,
      submissionPath: job.mode === 'code' ? path.join(jobDir, 'submission.py') : path.join(jobDir, datasetConfig.predictionFileName),
      benchmark: datasetConfig.benchmark,
      datasetPath: datasetConfig.datasetPath,
      datasetLabel: datasetConfig.datasetLabel,
      markersPath: datasetConfig.markersPath,
      trajectoryPath: datasetConfig.trajectoryPath,
      svgReferences: datasetConfig.svgReferences || {},
      hvgRange: datasetConfig.hvgRange,
      controlLabel: datasetConfig.controlLabel || 'ctrl',
      topK: datasetConfig.topK || 20,
      tasks: datasetConfig.tasks || null,
      outputPath: resultPath,
    };

    job.status = 'running';
    job.updatedAt = nowIso();
    job.progress = { stage: 'preparing', detail: 'Preparing evaluator workspace.' };
    await this.persistJob(job);

    const evaluatorTemplate = await readFile(this.templatePath, 'utf8');
    await writeFile(evaluatorPath, evaluatorTemplate, 'utf8');
    await writeFile(metadataPath, JSON.stringify(metadata, null, 2), 'utf8');

    job.progress = { stage: 'executing', detail: 'Running submission and evaluation pipeline.' };
    job.updatedAt = nowIso();
    await this.persistJob(job);

    const execution = await runCommand({
      command: pythonExecutable,
      args: [evaluatorPath, '--config', metadataPath],
      cwd: jobDir,
      timeoutMs: this.config.executionTimeoutSeconds * 1000,
      env: process.env,
    });

    job.logs = [execution.stdout, execution.stderr].filter(Boolean);

    if (!execution.ok) {
      job.status = 'failed';
      job.updatedAt = nowIso();
      job.error = execution.stderr || execution.stdout || 'Evaluation failed.';
      job.progress = { stage: 'failed', detail: 'Submission execution failed.' };
      await this.persistJob(job);
      return;
    }

    const rawResult = await loadJson(resultPath);
    const taskSummary = rawResult?.score_summary || {};
    job.status = rawResult?.status === 'failed' ? 'failed' : 'completed';
    job.updatedAt = nowIso();
    job.progress = { stage: job.status, detail: rawResult?.message || 'Evaluation complete.' };
    job.result = {
      ...rawResult,
      overallAverage: rawResult?.overall_average ?? scoreAverage(taskSummary),
    };
    job.error = rawResult?.status === 'failed' ? rawResult?.message || 'Evaluation failed.' : null;
    await this.persistJob(job);
  }

  async persistJob(job) {
    const jobDir = path.join(this.jobsDir, job.id);
    await ensureDir(jobDir);
    await writeFile(path.join(jobDir, 'job.json'), JSON.stringify(job, null, 2), 'utf8');
  }
}
