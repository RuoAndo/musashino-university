$env:Path = ($env:Path -split ';' | Where-Object { $_ -notmatch 'cygwin64\\bin' }) -join ';'
$env:PYTENSOR_FLAGS="cxx=,linker=py,device=cpu"
python -m jupyter lab