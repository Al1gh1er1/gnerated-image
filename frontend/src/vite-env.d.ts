/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly RUNPOD_API_KEY: string;
    readonly RUNPOD_ENDPOINT_ID: string;
  }
  
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
  