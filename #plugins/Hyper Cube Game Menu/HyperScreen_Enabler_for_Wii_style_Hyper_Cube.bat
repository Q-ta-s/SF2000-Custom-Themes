@ECHO OFF
@REM ===========================================================================
@REM =========== HyperScreen Informations ===========

SET QS_HYPER_LIST_VALUE=0 80 640 400

SET QS_DEF_SCREEN_VALUE=24 102 144 208

@REM ===========================================================================
SET QS_FOLDER_NAM_INI=Foldername.ini
SET QS_FOLDER_NAM_SYSTEM_START_NUM=4
SET QS_FOLDER_NAM_SYSTEM_NOT_NAM=ROMS
SET QS_FIND_EXE="%SystemRoot%\System32\FINDSTR.EXE"
IF NOT EXIST %QS_FIND_EXE% SET QS_FIND_EXE=FINDSTR

@REM ===========================================================================
CALL :CURRENT_DRIVE_CHK
IF NOT "%QS_CURRENT_DIR%" == "" SET QS_SD_DRIVE=%QS_CURRENT_DIR%&& GOTO :INPUT_CHECK1
IF NOT "%~1" == "" SET QS_SD_DRIVE=%~1&& GOTO :INPUT_CHECK1
@REM ===========================================================================
:INPUT_START

ECHO.
ECHO Please enter the drive or path where your SD card is located e.g. %QS_CURRENT_DRV%
SET QS_DEFAULT_MESS=""
IF NOT "%QS_CURRENT_DIR%" == "" SET QS_DEFAULT_MESS="(ENTER for Default: [93m%QS_CURRENT_DIR%[0m) "
SET QS_SD_DRIVE=
SET /P QS_SD_DRIVE=%QS_DEFAULT_MESS%
SET QS_SD_DRIVE=%QS_SD_DRIVE%*
SET QS_SD_DRIVE=%QS_SD_DRIVE:"=%
@REM IF "%QS_SD_DRIVE%" == "*" GOTO :INPUT_START
@REM SET QS_SD_DRIVE=%QS_SD_DRIVE:~0,-1%
IF "%QS_SD_DRIVE%" == "*" (
  SET QS_SD_DRIVE=%QS_CURRENT_DIR%
) ELSE (
  SET QS_SD_DRIVE=%QS_SD_DRIVE:~0,-1%
)
IF "%QS_SD_DRIVE%" == "" GOTO :INPUT_START

@REM ===========================================================================
:INPUT_CHECK1

ECHO "%QS_SD_DRIVE%" | %QS_FIND_EXE% ":" >NUL
IF ERRORLEVEL 1 SET QS_SD_DRIVE=%QS_SD_DRIVE%:

IF NOT EXIST "%QS_SD_DRIVE%" (
  ECHO ! Specified drive or path is not accessible
  GOTO :INPUT_START
)

IF NOT EXIST "%QS_SD_DRIVE%\Resources" (
  ECHO ! Couldn't find Resources folder "[93m%QS_SD_DRIVE%\Resources[0m"
  ECHO   Check the provided path points to an SD card!
  GOTO :INPUT_START
)

IF NOT EXIST "%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI%" (
IF NOT EXIST "%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI:e.ini=X.ini%" (
  ECHO ! Couldn't find "%QS_FOLDER_NAM_INI%" file "[93m%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI%[0m"
  ECHO   Check the provided path points to an SD card!
  GOTO :INPUT_START
)
)
:INPUT_CHECK1_BS_CHK
IF "%QS_SD_DRIVE:~-1%" == "\" (
  SET QS_SD_DRIVE=%QS_SD_DRIVE:~0,-1%
  GOTO :INPUT_CHECK1_BS_CHK
)

IF EXIST "%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI:e.ini=X.ini%" (
  @REM ********* 13 MENU MODE *******************
  SET QS_FOLDER_NAM_INI=FoldernamX.ini
)

CALL :GET_FOLDER_NAM "%QS_SD_DRIVE%"
IF ERRORLEVEL 1 (
  SET QS_FOLDER_NAM_INI=Foldername.ini
  GOTO :INPUT_START
)

REM ECHO === [[93m%QS_SD_DRIVE%[0m] was specified ===

@REM ===========================================================================
:START_HYPERSCREEN

CALL :CHECK_13_MODE

SETLOCAL ENABLEDELAYEDEXPANSION

SET "CRLF=<CRLF>"

SET QS_MODE_TYP=0
SET QS_LINE_NAM=0
FOR /F "usebackq delims=" %%A IN ("%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI%") DO (
  SET QS_LINE_BUF=%%A
  SET /A QS_LINE_NAM=!QS_LINE_NAM! + 1
  IF !QS_LINE_NAM! GTR %QS_FOLDER_NAM% (

    @REM *** FIND SCREEN DEFINE LINE ****** 
    FOR /F "tokens=1-4* delims= " %%B IN ("%%A") DO (
      CALL :NUMBER_CHK %%B
      IF NOT DEFINED QS_NUM (
        IF NOT "%%E"=="" IF "%%F"=="" (
          @REM *** FIND OK ¨ CHANGE SCREEN ******
          IF "!QS_LINE_BUF!"=="%QS_HYPER_LIST_VALUE%" (
            SET QS_MODE_TYP=1
            SET QS_LINE_BUF=%QS_DEF_SCREEN_VALUE%
          ) ELSE (
            SET QS_LINE_BUF=%QS_HYPER_LIST_VALUE%
          )
          @REM ***********************************
        )
      )
    )
    @REM **********************************
  )

  SET QS_RET=!QS_RET!!QS_LINE_BUF!!CRLF!
)

ECHO.
IF !QS_MODE_TYP!==0 (
  ECHO = [93mDo you want to enable [0m[36m"HyperScreen"[0m?
)
IF !QS_MODE_TYP!==1 (
  ECHO = HyperScreen is currently [93menabled[0m. Do you want to disable?
)
ECHO.
ECHO Push Enter to continue, or Type N to cancel
SET QS_SD_SC=
SET /P QS_SD_SC=""
IF "!QS_SD_SC!" == "n" GOTO :CANCEL_HYPERSCREEN
IF "!QS_SD_SC!" == "N" GOTO :CANCEL_HYPERSCREEN

IF !QS_MODE_TYP!==0 (
  COPY "%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI%" "%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI%_orig">NUL
  SET QS_DONE_MES=HyperScreen is now [36menabled[0m.
)
IF !QS_MODE_TYP!==1 (
  COPY "%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI%" "%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI%_hyper">NUL
  SET QS_DONE_MES=HyperScreen has been disabled.
)

POWERSHELL -NOPROFILE -EXECUTIONPOLICY BYPASS -COMMAND "$TEXT = '%QS_RET%'; $TEXT = $TEXT -REPLACE '<CRLF>', \"`r`n\"; [CONSOLE]::WRITE($TEXT)" > "%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI%"
IF ERRORLEVEL 1 (
  ENDLOCAL
  ECHO ! Write Error. "[93m%QS_SD_DRIVE%\Resources\%QS_FOLDER_NAM_INI%[0m"
  GOTO :CANCEL_HYPERSCREEN
)

CLS
ECHO.
ECHO !QS_DONE_MES!

ENDLOCAL
ECHO.
ECHO Press enter to exit
PAUSE>NUL
EXIT /B 0

:CANCEL_HYPERSCREEN

ECHO Cancelled.
ECHO Press enter to exit
PAUSE>NUL
EXIT /B

@REM ###########################################################################
:GET_FOLDER_NAM

SET QS_SYSTEM_LIST=
SET QS_FOLDER_NAM=0
SET QS_FOLDER_NAM_CMM=
SET QS_FOR_PARAM=usebackq skip=%QS_FOLDER_NAM_SYSTEM_START_NUM% tokens=1,2*
FOR /F "%QS_FOR_PARAM%" %%R IN ("%~1\Resources\%QS_FOLDER_NAM_INI%") DO (
  SET QS_FOR_SYS=
  IF "%%S" == "" (
    @REM ********* 13 MENU MODE *******************
    IF "%%R" == "" GOTO :GET_FOLDER_NAM_EXIT
    CALL :NUMBER_CHK %%R
    IF NOT DEFINED QS_NUM GOTO :GET_FOLDER_NAM_EXIT
    SET QS_FOR_SYS=%%R
    @REM ******************************************
  ) ELSE (
    IF "%%S" == "%QS_FOLDER_NAM_SYSTEM_NOT_NAM%" GOTO :GET_FOLDER_NAM_EXIT
    CALL :NUMBER_CHK %%S
    IF NOT DEFINED QS_NUM GOTO :GET_FOLDER_NAM_EXIT
    SET QS_FOR_SYS=%%S
    IF NOT "%%T" == "" SET QS_FOR_SYS=%%S %%T
  )
  CALL SET QS_SYSTEM_LIST=%%QS_SYSTEM_LIST%%%%QS_FOLDER_NAM_CMM%%%%QS_FOR_SYS%%
  CALL SET QS_SYSTEM_NAME[%%QS_FOLDER_NAM%%]=%%QS_FOR_SYS%%
  CALL SET /A QS_FOLDER_NAM=%%QS_FOLDER_NAM%% + 1
  @REM CALL ECHO %%QS_FOR_SYS%% - %%QS_FOLDER_NAM%%
  SET QS_FOLDER_NAM_CMM=, 
)

:GET_FOLDER_NAM_EXIT
IF NOT "%QS_SYSTEM_LIST%" == "" EXIT /B 0

ECHO ! SD card not read file. "[93m%~1\Resources\%QS_FOLDER_NAM_INI%[0m"
ECHO ! If [93m%QS_FOLDER_NAM_INI%[0m is open, close and try running again.
EXIT /B 1

@REM ###########################################################################
:CHECK_13_MODE

SET QS_13_MENU_MODE=0
IF %QS_FOLDER_NAM% GTR 8 (
  IF EXIST "%QS_SD_DRIVE%\Resources\m01.mm" GOTO :CHECK_13_MODE_EXIT
)
EXIT /B

:CHECK_13_MODE_EXIT
ECHO ***[93m [0m[36mNew menus extended version[0m[93m has been detected.[0m ***
SET QS_13_MENU_MODE=1
EXIT /B

@REM ###########################################################################
:CURRENT_DRIVE_CHK

PUSHD %~dp0\..\..

SET QS_CURRENT_DIR=%CD%
SET QS_CURRENT_DRV=
IF NOT EXIST "%QS_CURRENT_DIR%\Resources" (
  SET QS_CURRENT_DIR=*
) ELSE (
  IF EXIST "%QS_CURRENT_DIR%\Resources\%QS_FOLDER_NAM_INI%" GOTO :CURRENT_DRIVE_CHK_NEXT
  IF EXIST "%QS_CURRENT_DIR%\Resources\%QS_FOLDER_NAM_INI:e.ini=X.ini%" GOTO :CURRENT_DRIVE_CHK_NEXT
  SET QS_CURRENT_DIR=*
)
IF "%QS_CURRENT_DIR%" == "*" (
  IF EXIST "%~d0\Resources" (
    IF EXIST "%~d0\Resources\%QS_FOLDER_NAM_INI%" SET QS_CURRENT_DIR=%~d0
    IF EXIST "%~d0\Resources\%QS_FOLDER_NAM_INI:e.ini=X.ini%" SET QS_CURRENT_DIR=%~d0
  )
)
:CURRENT_DRIVE_CHK_NEXT
IF "%QS_CURRENT_DIR:~-1%" == "\" SET QS_CURRENT_DIR=%QS_CURRENT_DIR:~0,-1%
IF "%QS_CURRENT_DIR:~-1%" == ":" SET QS_CURRENT_DRV=%QS_CURRENT_DIR%
IF "%QS_CURRENT_DIR%" == "*" SET QS_CURRENT_DIR=
IF "%QS_CURRENT_DRV%" == "" SET QS_CURRENT_DRV=F:

POPD

EXIT /B

@REM ###########################################################################
:NUMBER_CHK

SET QS_NUM=%1

IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:0=%
IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:1=%
IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:2=%
IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:3=%
IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:4=%
IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:5=%
IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:6=%
IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:7=%
IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:8=%
IF DEFINED QS_NUM SET QS_NUM=%QS_NUM:9=%

EXIT /B
