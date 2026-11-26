import { createRawSnippet } from 'svelte';
import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import Layout from '../+layout.svelte';
import Privacy from './+page.svelte';

describe('/privacy', () => {
  it('states how uploads are handled, when it was updated, and how to get in touch', () => {
    const { body } = render(Privacy);

    expect(body).toContain('never written to disk, stored, logged, or used to train');
    expect(body).toContain('kept for 14 days');
    expect(body).toContain('No accounts, no cookies, no analytics');
    expect(body).toContain('datetime="2026-11-26"');
    expect(body).toContain('href="https://github.com/isaiah-harville/opendiagnostics/issues"');
  });

  it('is linked from the layout on every page', () => {
    const { body } = render(Layout, {
      props: {
        data: { catalog: { kind: 'unavailable' }, allowUploads: true },
        children: createRawSnippet(() => ({ render: () => '<p>page</p>' })),
      },
    });

    expect(body).toContain('href="/privacy"');
  });
});
