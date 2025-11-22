# ================================
# make-slideshow.ps1 (concat demuxer / glob非対応ビルド用)
# ================================

# --- Settings ---
$InputDir   = "nst_out"           # image folder
$OutMp4     = "nst_slideshow.mp4" # output mp4
$SecPerImg  = 2                   # seconds per image
$W          = 1280                # output width
$H          = 720                 # output height
$FpsOut     = 30                  # output fps
$Crf        = 18                  # quality (lower=better; 16-23)
$Preset     = "veryfast"          # encode speed

# --- Check ffmpeg ---
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg not found. Add it to PATH or use full path."
    exit 1
}

# --- Collect images (sorted) ---
$imgs = Get-ChildItem -Path $InputDir -Filter *.png -File | Sort-Object Name
if ($imgs.Count -eq 0) {
    Write-Error "No images found: $InputDir\*.png"
    exit 1
}

Write-Host ""
Write-Host "=== ffmpeg slideshow (concat) ==="
Write-Host "Input dir      : $InputDir"
Write-Host "Image count    : $($imgs.Count)"
Write-Host "Seconds/image  : $SecPerImg"
Write-Host "Output size    : ${W}x${H}"
Write-Host "Output file    : $OutMp4"
Write-Host "================================"
Write-Host ""

# --- Build concat list file ---
# Use forward slashes in paths for ffmpeg on Windows
$TmpList = Join-Path $env:TEMP ("ffconcat_" + [System.Guid]::NewGuid().ToString("N") + ".txt")
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("ffconcat version 1.0")

for ($i = 0; $i -lt $imgs.Count; $i++) {
    $p = $imgs[$i].FullName.Replace('\','/')
    # "file" line + per-file duration
    [void]$sb.AppendLine(("file '{0}'" -f $p))
    [void]$sb.AppendLine(("duration {0}" -f $SecPerImg))
}
# concat demuxer ignores duration for the last file unless repeated
$lastPath = $imgs[-1].FullName.Replace('\','/')
[void]$sb.AppendLine(("file '{0}'" -f $lastPath))

$sb.ToString() | Set-Content -Path $TmpList -Encoding ASCII

Write-Host "Concat list: $TmpList"
Write-Host "Running ffmpeg..."

# --- Video filter string (use ${} to avoid colon parsing issue) ---
$vf = "scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"

# --- Build args and run ffmpeg ---
$ffArgs = @(
    "-y",
    "-f","concat","-safe","0",
    "-i", $TmpList,
    "-vf", $vf,
    "-r", $FpsOut,
    "-c:v","libx264",
    "-crf", $Crf,
    "-preset", $Preset,
    "-movflags","+faststart",
    $OutMp4
)

& ffmpeg @ffArgs
$code = $LASTEXITCODE

if ($code -eq 0) {
    Write-Host ""
    Write-Host "Done: $OutMp4"
} else {
    Write-Error "ffmpeg failed with exit code $code."
}

# Optional: keep the list for debugging. Uncomment to delete after success.
# if ($code -eq 0) { Remove-Item $TmpList -ErrorAction SilentlyContinue }
