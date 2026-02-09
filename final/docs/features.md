# Dataset Feature Analysis

## Overview

- **Source**: `data/dataset.csv`
- **Total Rows**: 50,400
- **Total Columns**: 43
- **Domain**: Manufacturing facility sensor data
- **Temporal Resolution**: 1-minute intervals

## Temporal Range

- **Start**: 2025-01-13 00:00:00+00:00
- **End**: 2025-02-16 23:59:00+00:00
- **Span**: 34 days (839.98 hours)
- **Missing timestamps**: 0

## Data Quality Summary

| Column | Total | Valid | Missing | Missing % | Empty | N/A | Null | NaN |
|--------|-------|-------|---------|-----------|-------|-----|------|-----|
| DateTime | 50400 | 50400 | 0 | 0.0% | 0 | 0 | 0 | 0 |
| ActivePower | 50400 | 49385 | 1015 | 2.01% | 1015 | 0 | 0 | 0 |
| ReleaseAmountA | 50400 | 48141 | 2259 | 4.48% | 2259 | 0 | 0 | 0 |
| ReleaseAmountB | 50400 | 48279 | 2121 | 4.21% | 2121 | 0 | 0 | 0 |
| ReleaseAmountC | 50400 | 48231 | 2169 | 4.3% | 2169 | 0 | 0 | 0 |
| UpperRingRaiseB | 50400 | 48564 | 1836 | 3.64% | 1836 | 0 | 0 | 0 |
| UpperRingRaiseA | 50400 | 48564 | 1836 | 3.64% | 1836 | 0 | 0 | 0 |
| UpperRingRaiseC | 50400 | 48564 | 1836 | 3.64% | 1836 | 0 | 0 | 0 |
| UpperRingReleaseC | 50400 | 48564 | 1836 | 3.64% | 1836 | 0 | 0 | 0 |
| UpperRingReleaseB | 50400 | 48564 | 1836 | 3.64% | 1836 | 0 | 0 | 0 |
| UpperRingReleaseA | 50400 | 48564 | 1836 | 3.64% | 1836 | 0 | 0 | 0 |
| GasPressureUnderFurnaceA | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| GasPressureUnderFurnaceB | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| GasPressureUnderFurnaceC | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| MetalOutputIntensity | 50400 | 49133 | 1267 | 2.51% | 1267 | 0 | 0 | 0 |
| PowerB | 50400 | 48567 | 1833 | 3.64% | 1833 | 0 | 0 | 0 |
| PowerC | 50400 | 48567 | 1833 | 3.64% | 1833 | 0 | 0 | 0 |
| PowerA | 50400 | 48567 | 1833 | 3.64% | 1833 | 0 | 0 | 0 |
| HighVoltageA | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| HighVoltageB | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| HighVoltageC | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| LowerRingReleaseB | 50400 | 48564 | 1836 | 3.64% | 1836 | 0 | 0 | 0 |
| LowerRingReleaseC | 50400 | 48564 | 1836 | 3.64% | 1836 | 0 | 0 | 0 |
| LowerRingReleaseA | 50400 | 48564 | 1836 | 3.64% | 1836 | 0 | 0 | 0 |
| VentialtionValveForMantelA | 50400 | 44027 | 6373 | 12.64% | 6373 | 0 | 0 | 0 |
| VentialtionValveForMantelB | 50400 | 45785 | 4615 | 9.16% | 4615 | 0 | 0 | 0 |
| VentialtionValveForMantelC | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| ReactivePower | 50400 | 48567 | 1833 | 3.64% | 1833 | 0 | 0 | 0 |
| VoltageStepB | 50400 | 48249 | 2151 | 4.27% | 2151 | 0 | 0 | 0 |
| VoltageStepC | 50400 | 48556 | 1844 | 3.66% | 1844 | 0 | 0 | 0 |
| VoltageStepA | 50400 | 48553 | 1847 | 3.66% | 1847 | 0 | 0 | 0 |
| CurrentHolderPositionA | 50400 | 48552 | 1848 | 3.67% | 1848 | 0 | 0 | 0 |
| CurrentHolderPositionB | 50400 | 48538 | 1862 | 3.69% | 1862 | 0 | 0 | 0 |
| CurrentHolderPositionC | 50400 | 48548 | 1852 | 3.67% | 1852 | 0 | 0 | 0 |
| HolderModeB | 50400 | 48385 | 2015 | 4.0% | 2015 | 0 | 0 | 0 |
| HolderModeC | 50400 | 48331 | 2069 | 4.11% | 2069 | 0 | 0 | 0 |
| HolderModeA | 50400 | 48482 | 1918 | 3.81% | 1918 | 0 | 0 | 0 |
| AirTemperatureMantelA | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| AirTemperatureMantelB | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| AirTemperatureMantelC | 50400 | 49374 | 1026 | 2.04% | 1026 | 0 | 0 | 0 |
| FurnacePodTemparature | 50400 | 49373 | 1027 | 2.04% | 1027 | 0 | 0 | 0 |
| FurnaceBathTemperature | 50400 | 43779 | 6621 | 13.14% | 6621 | 0 | 0 | 0 |
| PowerSetpoint | 50400 | 46864 | 3536 | 7.02% | 3536 | 0 | 0 | 0 |

## Numeric Feature Statistics

| Column | Min | Max | Mean | Median | Std Dev | Unique |
|--------|-----|-----|------|--------|---------|--------|
| ActivePower | 0.0 | 38.054842 | 27.268114 | 28.712611 | 5.376272 | 1783 |
| ReleaseAmountA | 0.0 | 444.443604 | 175.798464 | 168.164062 | 112.406862 | 902 |
| ReleaseAmountB | 0.0 | 442.271484 | 147.101805 | 130.25708 | 110.838022 | 793 |
| ReleaseAmountC | 0.0 | 436.175537 | 148.271338 | 129.696533 | 106.576278 | 842 |
| UpperRingRaiseB | 0.0 | 1.0 | 0.002924 | 0.0 | 0.053995 | 2 |
| UpperRingRaiseA | 0.0 | 1.0 | 0.013323 | 0.0 | 0.114652 | 2 |
| UpperRingRaiseC | 0.0 | 1.0 | 0.003253 | 0.0 | 0.056946 | 2 |
| UpperRingReleaseC | 0.0 | 1.0 | 0.014043 | 0.0 | 0.117669 | 2 |
| UpperRingReleaseB | 0.0 | 1.0 | 0.003048 | 0.0 | 0.05512 | 2 |
| UpperRingReleaseA | 0.0 | 1.0 | 0.014682 | 0.0 | 0.120275 | 2 |
| GasPressureUnderFurnaceA | -168.75 | 168.75 | -39.392596 | -20.629677 | 60.490943 | 9422 |
| GasPressureUnderFurnaceB | -227.8125 | 227.8125 | -55.4272 | -31.238789 | 85.575684 | 9385 |
| GasPressureUnderFurnaceC | -125.0 | 125.0 | -33.789525 | -20.96875 | 45.098319 | 9472 |
| MetalOutputIntensity | 0.0 | 22.4 | 6.840808 | 0.0 | 9.486709 | 473 |
| PowerB | 0.0 | 19.835735 | 10.699623 | 11.066115 | 2.58279 | 47872 |
| PowerC | 0.0 | 17.694487 | 11.833708 | 12.519854 | 2.792494 | 47798 |
| PowerA | 0.0 | 19.709267 | 11.420915 | 12.199585 | 2.915391 | 47856 |
| HighVoltageA | 142.725 | 194.60078 | 186.437666 | 186.836542 | 4.05308 | 766 |
| HighVoltageB | 142.725 | 192.811953 | 185.206818 | 185.523467 | 3.705346 | 702 |
| HighVoltageC | 142.725 | 191.289558 | 182.406309 | 182.70703 | 3.975716 | 679 |
| LowerRingReleaseB | 0.0 | 1.0 | 0.006754 | 0.0 | 0.081905 | 2 |
| LowerRingReleaseC | 0.0 | 1.0 | 0.007207 | 0.0 | 0.084587 | 2 |
| LowerRingReleaseA | 0.0 | 1.0 | 0.007392 | 0.0 | 0.08566 | 2 |
| VentialtionValveForMantelA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| VentialtionValveForMantelB | 194.112488 | 200.0 | 199.118473 | 200.0 | 1.479679 | 468 |
| VentialtionValveForMantelC | 9.387499 | 10.1125 | 9.675871 | 9.649998 | 0.147013 | 59 |
| ReactivePower | 0.0 | 42.960883 | 12.303512 | 12.207661 | 3.193142 | 1843 |
| VoltageStepB | 0.0 | 41.4 | 38.830504 | 39.622948 | 3.066708 | 221 |
| VoltageStepC | 0.0 | 41.4 | 36.536994 | 36.046575 | 3.081172 | 195 |
| VoltageStepA | 0.0 | 41.4 | 37.998479 | 37.786501 | 3.184552 | 97 |
| CurrentHolderPositionA | -7.59375 | 678.867188 | 170.438415 | 128.109375 | 143.809898 | 7100 |
| CurrentHolderPositionB | -118.125 | 991.6875 | 217.569281 | 163.828125 | 188.878494 | 8739 |
| CurrentHolderPositionC | -9.140625 | 752.484375 | 241.656145 | 208.96875 | 175.991352 | 8279 |
| HolderModeB | 0.0 | 1.0 | 0.917433 | 1.0 | 0.275226 | 2 |
| HolderModeC | 0.0 | 1.0 | 0.924458 | 1.0 | 0.264263 | 2 |
| HolderModeA | 0.0 | 1.0 | 0.920321 | 1.0 | 0.270796 | 2 |
| AirTemperatureMantelA | 28.492887 | 249.808472 | 101.609546 | 98.346379 | 30.712678 | 3115 |
| AirTemperatureMantelB | 43.295625 | 433.391617 | 188.259686 | 185.663234 | 46.071575 | 4956 |
| AirTemperatureMantelC | 18.237369 | 335.819266 | 51.851908 | 49.753681 | 17.010196 | 1712 |
| FurnacePodTemparature | 664.36875 | 701.775 | 683.920446 | 682.931305 | 11.577155 | 434 |
| FurnaceBathTemperature | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| PowerSetpoint | 0.0 | 48.244 | 41.935024 | 43.075 | 3.182782 | 14 |

## Binary Features (0/1)

These columns contain only 0 and 1 values, likely representing on/off states or flags.

| Column | Zero Count | Zero % | One Count | One % |
|--------|------------|--------|-----------|-------|
| UpperRingRaiseB | 48422 | 99.71% | 142 | 0.29% |
| UpperRingRaiseA | 47917 | 98.67% | 647 | 1.33% |
| UpperRingRaiseC | 48406 | 99.67% | 158 | 0.33% |
| UpperRingReleaseC | 47882 | 98.6% | 682 | 1.4% |
| UpperRingReleaseB | 48416 | 99.7% | 148 | 0.3% |
| UpperRingReleaseA | 47851 | 98.53% | 713 | 1.47% |
| LowerRingReleaseB | 48236 | 99.32% | 328 | 0.68% |
| LowerRingReleaseC | 48214 | 99.28% | 350 | 0.72% |
| LowerRingReleaseA | 48205 | 99.26% | 359 | 0.74% |
| VentialtionValveForMantelA | 44027 | 100.0% | 0 | 0.0% |
| HolderModeB | 3995 | 8.26% | 44390 | 91.74% |
| HolderModeC | 3651 | 7.55% | 44680 | 92.45% |
| HolderModeA | 3863 | 7.97% | 44619 | 92.03% |
| FurnaceBathTemperature | 43779 | 100.0% | 0 | 0.0% |

## High-Zero Columns (>50% zeros, non-binary)

These columns have a high percentage of zero values, suggesting sparse activity or events.

| Column | Zeros Count | Zeros % | Min | Max | Mean |
|--------|-------------|---------|-----|-----|------|
| MetalOutputIntensity | 29137 | 59.3% | 0.0 | 22.4 | 6.840808 |

## Columns with Negative Values

| Column | Negative Count | Negative % | Min | Max |
|--------|----------------|------------|-----|-----|
| GasPressureUnderFurnaceA | 35678 | 72.26% | -168.75 | 168.75 |
| GasPressureUnderFurnaceB | 39171 | 79.34% | -227.8125 | 227.8125 |
| GasPressureUnderFurnaceC | 40147 | 81.31% | -125.0 | 125.0 |
| CurrentHolderPositionA | 628 | 1.29% | -7.59375 | 678.867188 |
| CurrentHolderPositionB | 1261 | 2.6% | -118.125 | 991.6875 |
| CurrentHolderPositionC | 1160 | 2.39% | -9.140625 | 752.484375 |

## Column Descriptions

### Process Columns
| Column | Description |
|--------|-------------|
| DateTime | Timestamp of the measurement (ISO 8601 format, UTC) |
| ActivePower | Active electrical power of the furnace (MW) |
| ReactivePower | Reactive electrical power (MVAr) |
| PowerSetpoint | Target power setpoint for the furnace |
| PowerA / PowerB / PowerC | Per-phase power measurements |
| HighVoltageA / HighVoltageB / HighVoltageC | High voltage per phase |
| VoltageStepA / VoltageStepB / VoltageStepC | Voltage step/tap position per phase |

### Release & Raise Columns
| Column | Description |
|--------|-------------|
| ReleaseAmountA / B / C | Material release amounts per phase |
| UpperRingRaiseA / B / C | Upper ring raise flags (binary) |
| UpperRingReleaseA / B / C | Upper ring release flags (binary) |
| LowerRingReleaseA / B / C | Lower ring release flags (binary) |

### Gas Pressure & Temperature Columns
| Column | Description |
|--------|-------------|
| GasPressureUnderFurnaceA / B / C | Gas pressure under furnace per phase (negative values = vacuum) |
| AirTemperatureMantelA / B / C | Air temperature in the furnace mantel per phase |
| FurnacePodTemparature | Furnace pod temperature |
| FurnaceBathTemperature | Furnace bath temperature |

### Holder & Ventilation Columns
| Column | Description |
|--------|-------------|
| CurrentHolderPositionA / B / C | Current electrode holder position per phase |
| HolderModeA / B / C | Holder operation mode (binary: 0 or 1) |
| VentialtionValveForMantelA / B / C | Ventilation valve opening for mantel per phase |

### Production Columns
| Column | Description |
|--------|-------------|
| MetalOutputIntensity | Intensity of metal output from the furnace |

## Key Observations

1. **Missing values present as empty strings**: All missing data appears as empty CSV fields (not N/A, null, or NaN). Missing rates range from 2.01% (ActivePower) to 13.14% (FurnaceBathTemperature). The highest missing rates are: FurnaceBathTemperature (13.14%), VentialtionValveForMantelA (12.64%), VentialtionValveForMantelB (9.16%), and PowerSetpoint (7.02%).
2. **Two constant/dead columns**: `FurnaceBathTemperature` is always 0.0 (with 13.14% missing) and `VentialtionValveForMantelA` is always 0.0 (with 12.64% missing). Both are candidates for removal as they carry no information.
3. **Binary indicator columns**: UpperRingRaise A/B/C, UpperRingRelease A/B/C, LowerRingRelease A/B/C, and HolderMode A/B/C are binary flags (0/1). The ring-related flags are extremely sparse (~99% zeros), while HolderMode is predominantly 1 (~92%).
4. **Gas pressure values are predominantly negative**: GasPressureUnderFurnace A/B/C have 72-81% negative values, indicating vacuum/suction conditions under the furnace. They can also swing positive.
5. **MetalOutputIntensity is highly sparse**: 59.3% zeros with a median of 0.0, suggesting metal output is intermittent rather than continuous.
6. **Three-phase symmetry**: Many features come in A/B/C variants, representing three-phase electrical measurements typical of electric arc furnaces.
7. **Holder positions can be negative**: CurrentHolderPosition B has values as low as -118.125, which may indicate retracted positions or calibration offsets.
8. **Temporal granularity**: Data is recorded at 1-minute intervals over 35 days, providing high-resolution process monitoring.
9. **Typos in column names**: `VentialtionValveForMantel` (should be Ventilation), `FurnacePodTemparature` (should be Temperature).
