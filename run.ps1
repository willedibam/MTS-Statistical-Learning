param(
  [ValidateSet('dev','paper')] [string]\ = 'dev',
  [ValidateSet('compute','visualize')] [string]\ = 'compute',
  [string]\ = '.\pyspi\configs\pilot0_config.yaml'
)
Write-Host "Running \ in \ with \"
.\.venv\Scripts\python.exe -m pyspi \ --mode \ --config \
