#! /usr/bin/pwsh
param(
    [parameter(Mandatory=$true)][string]$resourceGroup,
    [parameter(Mandatory=$true)][string]$location,
    [parameter(Mandatory=$true)][string]$subscription
)

Write-Host "Login in your account" -ForegroundColor Yellow
az login

Write-Host "Choosing your subscription" -ForegroundColor Yellow
az account set --subscription $subscription

Push-Location $($MyInvocation.InvocationName | Split-Path)
Push-Location Scripts

. "./Deploy-ARM.ps1"
DeployARM -resourceGroup $resourceGroup -location $location

Pop-Location
Pop-Location