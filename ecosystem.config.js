const baseAppConfigs = {
  namespace: 'default',
  interpreter: './venv/bin/python',
  env: {
    'PYTHONPATH': './'
  },
  watch: false,
  autorestart: false
};

module.exports = {
  apps: [
    {
      ...baseAppConfigs,
      name: 'Quick-Project-Celebrity-lookalike-Platform-Api',
      script: './apps/platform/src/app.py'
    },
    {
      ...baseAppConfigs,
      name: 'Quick-Project-Celebrity-lookalike-Platform-WebApp',
      script: './apps/platform_web/src/app.py'
    }
  ]
};
