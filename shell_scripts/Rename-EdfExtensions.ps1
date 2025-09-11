# PowerShell script to rename all .EDF files to .edf in the specified directory
# Usage: .\Rename-EdfExtensions.ps1

# Set the target directory
$TargetDir = "/work/jacobs_lab/EEG_Data/Clustering_Patho_HFO"

# Check if directory exists
if (-not (Test-Path -Path $TargetDir)) {
    Write-Error "Directory $TargetDir does not exist!"
    exit 1
}

Write-Host "Renaming .EDF files to .edf in: $TargetDir" -ForegroundColor Green
Write-Host "----------------------------------------" -ForegroundColor Green

# Get all .EDF files
$edfFiles = Get-ChildItem -Path $TargetDir -Filter "*.EDF" -Recurse

if ($edfFiles.Count -eq 0) {
    Write-Host "No .EDF files found in $TargetDir" -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($edfFiles.Count) .EDF files to rename" -ForegroundColor Cyan
Write-Host ""

# Counter for renamed files
$successCount = 0
$skipCount = 0
$errorCount = 0

# Process each file
foreach ($file in $edfFiles) {
    # Create new filename with .edf extension
    $newName = $file.BaseName + ".edf"
    $newPath = Join-Path -Path $file.Directory.FullName -ChildPath $newName
    
    # Check if target file already exists
    if (Test-Path -Path $newPath) {
        Write-Warning "File $newName already exists in $($file.Directory.Name). Skipping $($file.Name)"
        $skipCount++
    }
    else {
        try {
            # Rename the file
            Rename-Item -Path $file.FullName -NewName $newName -ErrorAction Stop
            Write-Host "Renamed: $($file.Name) -> $newName" -ForegroundColor White
            $successCount++
        }
        catch {
            Write-Error "Failed to rename $($file.Name): $($_.Exception.Message)"
            $errorCount++
        }
    }
}

Write-Host ""
Write-Host "Renaming complete!" -ForegroundColor Green
Write-Host "Successfully renamed: $successCount files" -ForegroundColor Green
Write-Host "Skipped (already exists): $skipCount files" -ForegroundColor Yellow
Write-Host "Errors: $errorCount files" -ForegroundColor Red
Write-Host "Total files processed: $($edfFiles.Count)" -ForegroundColor Cyan
