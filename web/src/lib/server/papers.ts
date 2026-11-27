import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parse } from 'yaml';
import { z } from 'zod';

const paperRecordSchema = z.object({
  title: z.string().min(1),
  description: z.string().min(1),
  project: z.string().min(1),
  run: z.string().min(1),
  pdf: z.string().regex(/^[\w.-]+\.pdf$/),
});

export type Paper = {
  name: string;
  title: string;
  description: string;
  project: string;
  run: string;
};

const NAME = /^[a-z0-9][a-z0-9-]*$/;

/** The papers in `dir`, one per subfolder with a `paper.yaml`, sorted by name. */
export async function listPapers(dir: string): Promise<Paper[]> {
  let names: string[];
  try {
    names = (await readdir(dir, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory() && NAME.test(entry.name))
      .map((entry) => entry.name)
      .sort();
  } catch (cause) {
    if ((cause as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw cause;
  }
  const papers: Paper[] = [];
  for (const name of names) {
    let text: string;
    try {
      text = await readFile(join(dir, name, 'paper.yaml'), 'utf8');
    } catch {
      continue;
    }
    const { title, description, project, run } = paperRecordSchema.parse(parse(text));
    papers.push({ name, title, description, project, run });
  }
  return papers;
}

export type PaperFile = 'pdf' | 'preview';

/** The bytes of a paper's PDF or first-page preview, or null when there is no such paper. */
export async function readPaperFile(
  dir: string,
  name: string,
  file: PaperFile,
): Promise<Uint8Array<ArrayBuffer> | null> {
  if (!NAME.test(name)) return null;
  try {
    const record = paperRecordSchema.parse(
      parse(await readFile(join(dir, name, 'paper.yaml'), 'utf8')),
    );
    return new Uint8Array(
      await readFile(join(dir, name, file === 'pdf' ? record.pdf : 'preview.png')),
    );
  } catch (cause) {
    if ((cause as NodeJS.ErrnoException).code === 'ENOENT') return null;
    throw cause;
  }
}
