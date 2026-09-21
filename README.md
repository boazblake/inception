# Inception

Minimal browser pose canvas using standalone MediaPipe Pose, Hand, and Face Landmarkers.

## Local development

```sh
npm ci
npm run dev
```

The camera requires a secure context (`localhost` is allowed) and permission to access the camera.

## Verification

```sh
npm run build
```

## GitHub Pages

Pushes to `master` build and deploy `dist` through GitHub Actions. Enable **GitHub Pages → Source: GitHub Actions** in the repository settings once, then the deployed site will be available at:

`https://boazblake.github.io/inception/`
