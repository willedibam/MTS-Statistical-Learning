param(
  [ValidateSet('dev','paper')] [string]$mode = 'dev',
  [ValidateSet('compute','visualize')] [string]$cmd = 'compute',
  [string]$config = '.\src\spimts\configs\pilot0_config.yaml'
)
python -m spimts $cmd --mode $mode --config $config
