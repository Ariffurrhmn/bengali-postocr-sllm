$ErrorActionPreference = 'Stop'
$DS = 'E:\.shortcut-targets-by-id\1KvOyKAwUxnGGzgVJioFCK_QhW7OfPHoL\Group 2\Datasets'
$OUT = "$DS\revision_2026-11"

$utf8 = New-Object System.Text.UTF8Encoding $false

function Norm($b) { (($b.ToLowerInvariant() -replace '[\. \-]', '_') -replace '_+', '_').Trim('_') }
function Core($b) { $b -replace '^(vt|t)_?', '' }

# ---------- pages ----------
$reid = Import-Csv "$DS\REID2019\manifest_all81.csv" -Encoding UTF8
$bl = Import-Csv "$DS\BL_Transkribus_manifest.csv" -Encoding UTF8
$pages = New-Object System.Collections.Generic.List[object]
foreach ($r in $reid) {
  $xmlName = [IO.Path]::GetFileNameWithoutExtension($r.tif_name)
  $x = Get-ChildItem -LiteralPath "$DS\REID2019\original" -Filter '*.xml' | Where-Object { $_.BaseName -ieq $r.page_id } | Select-Object -First 1
  [xml]$d = [IO.File]::ReadAllText($x.FullName)
  $u = $d.SelectNodes("//*[local-name()='TextRegion']/*[local-name()='TextEquiv']/*[local-name()='Unicode']")
  if ($u.Count -eq 0) { $u = $d.SelectNodes("//*[local-name()='TextLine']/*[local-name()='TextEquiv']/*[local-name()='Unicode']") }
  $txt = (($u | ForEach-Object { $_.InnerText }) -join ' ') -replace '[\r\n]+', ' '
  $pages.Add([pscustomobject]@{ page_id = $r.page_id; source = 'REID'; book = Norm $r.book_id; reid_split = $r.split
    gt_chars = $txt.Length; bfrac = 1.0; image_path = "REID2019\original\$($r.tif_name)"; gt_path = "REID2019\original\$($x.Name)" })
}
foreach ($b in $bl) {
  $pages.Add([pscustomobject]@{ page_id = $b.page_id; source = 'BL'; book = Norm $b.book_id; reid_split = ''
    gt_chars = [int]$b.gt_chars; bfrac = [double]$b.bengali_char_frac
    image_path = "BL_Transkribus\Transkribus_dataset_ImagesALTOXML\$($b.image_name)"; gt_path = "BL_Transkribus\Transkribus_dataset_ImagesALTOXML\$($b.page_id).xml" })
}
$dupIds = $pages | Group-Object { $_.page_id.ToLowerInvariant() } | Where-Object Count -gt 1
if ($dupIds) { throw "page_id collision: $($dupIds.Name -join ',')" }

# ---------- union-find over books ----------
$parent = @{}
function Find($a) { while ($parent[$a] -ne $a) { $parent[$a] = $parent[$parent[$a]]; $a = $parent[$a] }; $a }
function Union($a, $b, $why) { $ra = Find $a; $rb = Find $b; if ($ra -ne $rb) { $parent[$rb] = $ra; $script:merges.Add("$a + $b ($why)") } }
$script:merges = New-Object System.Collections.Generic.List[string]
$books = $pages | ForEach-Object book | Sort-Object -Unique
foreach ($k in $books) { $parent[$k] = $k }
# explicit aliases
$aliases = @(@('14129_b_1_12', '14129_b_12'), @('vt_280', 'vt_280_a'), @('vt_1712_d', 'vt1712_schoolbook'), @('vt_1712_d', 'vt_1712'), @('vt_1328', 'vt_1328_c'), @('vt_1355', 'vt_1355_a'))
foreach ($p in $aliases) { foreach ($k in $p) { if (-not $parent.ContainsKey($k)) { throw "alias book not found: $k" } }; Union $p[0] $p[1] 'alias' }
# exact / probable / sibling rules, all pairs
for ($i = 0; $i -lt $books.Count; $i++) { for ($j = $i + 1; $j -lt $books.Count; $j++) {
  $a = Core $books[$i]; $b = Core $books[$j]; $why = $null
  if ($a -eq $b) { $why = 'same after prefix strip' }
  elseif ($a.StartsWith("${b}_") -or $b.StartsWith("${a}_")) { $why = 'parent/sub shelfmark' }
  else {
    $pa = $a -split '_'; $pb = $b -split '_'
    # v2 strict sibling rule: same shelfmark after dropping the LAST component (any length).
    # Applied to the prefix-stripped core; the remaining parent must be non-empty, so single-component
    # cores such as vt_1328 -> '1328' never match each other.
    if ($pa.Count -ge 2 -and $pb.Count -ge 2 -and (($pa[0..($pa.Count - 2)] -join '_') -eq ($pb[0..($pb.Count - 2)] -join '_'))) { $why = 'sibling' }
  }
  if ($why) { Union $books[$i] $books[$j] $why }
} }
$groupOf = @{}; foreach ($k in $books) { $groupOf[$k] = Find $k }
# readable group id = alphabetically first book in group
$members = $books | Group-Object { $groupOf[$_] }
$gname = @{}; foreach ($m in $members) { $n = 'g_' + ($m.Group | Sort-Object | Select-Object -First 1); foreach ($k in $m.Group) { $gname[$k] = $n } }
foreach ($p in $pages) { $p | Add-Member group_id $gname[$p.book]; $p | Add-Member split ''; $p | Add-Member reason '' }
$reidGroups = @{}; foreach ($p in ($pages | Where-Object source -eq 'REID')) { $reidGroups[$p.group_id] = $true }

# ---------- assign ----------
foreach ($p in $pages) {
  if ($p.page_id -eq 'VT_1712_0090') { $p.split = 'unused'; $p.reason = 'byte-identical duplicate of REID eval page VT_1712_d_0001'; continue }
  if ($p.source -eq 'REID') {
    if ($p.reid_split -in 'eval', 'dev', 'dropped_thin') { $p.split = 'pool' } else { $p.split = 'unused'; $p.reason = 'REID page without ground truth' }
    continue
  }
  if (-not $reidGroups.ContainsKey($p.group_id)) {
    if ($p.gt_chars -ge 300 -and $p.bfrac -ge 0.5) { $p.split = 'test' }
    else { $p.split = 'unused'; $p.reason = "book-disjoint BL page below test threshold (gt_chars=$($p.gt_chars), bengali_frac=$($p.bfrac))" }
  } else {
    if ($p.gt_chars -ge 150) { $p.split = 'pool' } else { $p.split = 'unused'; $p.reason = "BL page from REID-linked group with gt_chars=$($p.gt_chars) < 150" }
  }
}

# ---------- val selection ----------
$pool = $pages | Where-Object split -eq 'pool'
$poolChars = ($pool | Measure-Object gt_chars -Sum).Sum
$gids = @($pool | ForEach-Object group_id | Sort-Object -Unique)
$rng = New-Object System.Random 2026
for ($i = $gids.Count - 1; $i -gt 0; $i--) { $j = $rng.Next($i + 1); $t = $gids[$i]; $gids[$i] = $gids[$j]; $gids[$j] = $t }
$valG = @{}; $valChars = 0; $skipped = @()
foreach ($g in $gids) {
  if ($valChars -ge 0.18 * $poolChars) { break }
  $c = ($pool | Where-Object group_id -eq $g | Measure-Object gt_chars -Sum).Sum
  if ($valChars + $c -le 0.22 * $poolChars) { $valG[$g] = $true; $valChars += $c } else { $skipped += "$g ($c chars, would overshoot 22%)" }
}
foreach ($p in $pool) { $p.split = $(if ($valG.ContainsKey($p.group_id)) { 'val' } else { 'train' }) }

# ---------- write ----------
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
function W($name, $lines) { [IO.File]::WriteAllLines("$OUT\$name", [string[]]$lines, $utf8) }
foreach ($s in 'test', 'train', 'val') {
  $f = $(if ($s -eq 'test') { 'test_bl.txt' } else { "$s.txt" })
  W $f ($pages | Where-Object split -eq $s | Sort-Object source, page_id | ForEach-Object page_id)
}
W 'unused.txt' ($pages | Where-Object split -eq 'unused' | Sort-Object source, page_id | ForEach-Object { "$($_.page_id)`t$($_.source)`t$($_.reason)" })
$csv = $pages | Sort-Object split, source, page_id | Select-Object page_id, source, group_id, split, gt_chars, @{n = 'bengali_char_frac'; e = { if ($_.source -eq 'BL') { $_.bfrac } else { '' } } }, @{n = 'old_reid_split'; e = { $_.reid_split } }, @{n = 'source_path'; e = { $_.image_path } }, @{n = 'gt_path'; e = { $_.gt_path } }, reason
W 'split_manifest.csv' ($csv | ConvertTo-Csv -NoTypeInformation)
W 'group_merges.txt' $script:merges

# ---------- summary ----------
$SUM = New-Object System.Collections.Generic.List[string]
$SUM.Add("v2, 2026-11-20, strict sibling rule chosen by the author before any test result (v1 kept in superseded_v1\)")
$SUM.Add("Split built $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') by make_splits.ps1 (seed 2026, val target 18-22% of pool GT chars)")
$SUM.Add("gt_chars: BL = ALTO String CONTENT words joined by spaces per line, lines joined without newline; REID = PAGE TextRegion TextEquiv/Unicode (TextLine fallback), newlines replaced by a space.")
$SUM.Add('')
foreach ($s in 'test', 'train', 'val', 'unused') {
  $g = $pages | Where-Object split -eq $s
  $SUM.Add(("{0,-7} pages={1,3}  groups={2,3}  gt_chars={3,7}  (REID {4} pages, BL {5} pages)" -f $s, @($g).Count, @($g | ForEach-Object group_id | Sort-Object -Unique).Count, ($g | Measure-Object gt_chars -Sum).Sum, @($g | Where-Object source -eq 'REID').Count, @($g | Where-Object source -eq 'BL').Count))
}
$SUM.Add(("val share of pool chars = {0:P1}" -f ($valChars / $poolChars)))
if ($skipped) { $SUM.Add("groups skipped during val fill: " + ($skipped -join '; ')) }
$SUM.Add('')
$leak = $pages | Where-Object split -in 'test', 'train', 'val' | Group-Object group_id | Where-Object { @($_.Group | ForEach-Object split | Sort-Object -Unique).Count -gt 1 }
$SUM.Add("CHECK no group in two splits (test/train/val): " + $(if ($leak) { 'FAIL ' + ($leak.Name -join ',') } else { 'PASS' }))
$testReid = $pages | Where-Object { $_.split -eq 'test' -and $reidGroups.ContainsKey($_.group_id) }
$SUM.Add("CHECK no test group contains any REID page (incl. no-GT pages): " + $(if ($testReid) { 'FAIL' } else { 'PASS' }))
$SUM.Add("CHECK VT_1712_0090 excluded: " + $(if (($pages | Where-Object page_id -eq 'VT_1712_0090').split -eq 'unused') { 'PASS' } else { 'FAIL' }))
$SUM.Add('')
$SUM.Add('Val groups: ' + (($valG.Keys | Sort-Object) -join ', '))
$SUM.Add('Test groups: ' + ((($pages | Where-Object split -eq 'test' | ForEach-Object group_id | Sort-Object -Unique)) -join ', '))
$SUM.Add('Multi-book groups: ' + (($members | Where-Object Count -gt 1 | ForEach-Object { $gname[$_.Group[0]] + '=[' + (($_.Group | Sort-Object) -join ' ') + ']' }) -join '; '))
$SUM.Add('')
$SUM.Add('SHA-256 of list files at freeze time:')
foreach ($f in 'test_bl.txt', 'train.txt', 'val.txt') { $SUM.Add("  $f  " + (Get-FileHash -LiteralPath "$OUT\$f" -Algorithm SHA256).Hash.ToLowerInvariant()) }
W 'split_summary.txt' $SUM
$SUM | ForEach-Object { $_ }
