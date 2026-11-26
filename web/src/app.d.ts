import type { ApiClient } from '#lib/server/api.ts';
import type { Config } from '#lib/server/config.ts';

declare global {
  namespace App {
    interface Locals {
      config: Config;
      api: ApiClient;
    }
  }
}
