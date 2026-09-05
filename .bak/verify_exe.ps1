$exe = 'E:\xiaomubiao\pyt\dist\PixEnhance_AI_画质超分大师.exe'
if (-not (Test-Path $exe)) { Write-Error "EXE not found: $exe"; exit 1 }

$bytes = [System.IO.File]::ReadAllBytes($exe)
$len = $bytes.Length
Write-Host "EXE: $exe"
Write-Host ("Size: {0:N0} bytes ({1:N1} KB)" -f $len, $len/1024)
Write-Host ""

$peOff = [BitConverter]::ToInt32($bytes, 0x3C)
$icoHits = 0
for ($i=0; $i -le $bytes.Length-4; $i++) {
    if ($bytes[$i] -eq 0x00 -and $bytes[$i+1] -eq 0x00 -and $bytes[$i+2] -eq 0x01 -and $bytes[$i+3] -eq 0x00) {
        $icoHits++
    }
}
Write-Host ("PE header offset: 0x{0:X}" -f $peOff)
Write-Host ("ICONDIR signature hits: $icoHits")
Write-Host ""

$needles = @(
    'realesrgan-ncnn-vulkan.exe',
    'vcomp140.dll',
    'engine/LICENSE',
    'engine/README.md',
    'PYZ-00.pyz'
)
Write-Host "Embedded resource scan:"
foreach ($n in $needles) {
    $b = [System.Text.Encoding]::UTF8.GetBytes($n)
    $found = $false
    for ($i=0; $i -le $bytes.Length-$b.Length; $i++) {
        $match = $true
        for ($j=0; $j -lt $b.Length; $j++) {
            if ($bytes[$i+$j] -ne $b[$j]) { $match = $false; break }
        }
        if ($match) { $found = $true; break }
    }
    $tag = if ($found) { '[OK]' } else { '[MISSING]' }
    Write-Host ("  {0,-28} {1}" -f $n, $tag)
}

$pngHits = 0
for ($i=0; $i -le $bytes.Length-4; $i++) {
    if ($bytes[$i] -eq 0x89 -and $bytes[$i+1] -eq 0x50 -and $bytes[$i+2] -eq 0x4E -and $bytes[$i+3] -eq 0x47) {
        $pngHits++
    }
}
Write-Host ""
Write-Host ("PNG signature 89504E47 hits: $pngHits")
