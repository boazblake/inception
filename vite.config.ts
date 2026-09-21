import { defineConfig } from "vite";

const repositoryName = process.env.GITHUB_REPOSITORY?.split("/").pop();
const base = process.env.GITHUB_ACTIONS && repositoryName
  ? `/${repositoryName}/`
  : "/";

export default defineConfig({
  base,
  build: {
    outDir: "dist",
  },
});
