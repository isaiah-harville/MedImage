import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { listPapers, readPaperFile } from './papers.ts';

async function papersDir() {
  const dir = await mkdtemp(join(tmpdir(), 'papers-'));
  await mkdir(join(dir, 'dermamnist'));
  await writeFile(
    join(dir, 'dermamnist', 'paper.yaml'),
    'title: A paper\ndescription: What it shows.\nproject: dermamnist-convnext\nrun: r1\npdf: main.pdf\n',
  );
  await writeFile(join(dir, 'dermamnist', 'main.pdf'), '%PDF-1.5');
  await writeFile(join(dir, 'dermamnist', 'preview.png'), 'png');
  await mkdir(join(dir, 'drafts-without-yaml'));
  return dir;
}

describe('papers', () => {
  it('lists each folder with a paper.yaml', async () => {
    expect(await listPapers(await papersDir())).toEqual([
      {
        name: 'dermamnist',
        title: 'A paper',
        description: 'What it shows.',
        project: 'dermamnist-convnext',
        run: 'r1',
      },
    ]);
  });

  it('reads the PDF and the preview of a listed paper', async () => {
    const dir = await papersDir();
    const pdf = await readPaperFile(dir, 'dermamnist', 'pdf');
    expect(new TextDecoder().decode(pdf ?? undefined)).toBe('%PDF-1.5');
    expect(await readPaperFile(dir, 'dermamnist', 'preview')).not.toBeNull();
  });

  it('returns nothing for a missing paper or a name that leaves the folder', async () => {
    const dir = await papersDir();
    expect(await readPaperFile(dir, 'nope', 'pdf')).toBeNull();
    expect(await readPaperFile(dir, '../etc', 'pdf')).toBeNull();
  });

  it('has no papers when the folder does not exist', async () => {
    expect(await listPapers(join(tmpdir(), 'no-such-papers-folder'))).toEqual([]);
  });
});
