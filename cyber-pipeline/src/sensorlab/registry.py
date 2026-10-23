"""Stable public sensor-to-station mapping; no provisioned security identities."""

from types import MappingProxyType

SENSOR_STATIONS = MappingProxyType(
    {
        "uscrn-az-tucson-11-w": "AZ_Tucson_11_W",
        "uscrn-ca-fallbrook-5-ne": "CA_Fallbrook_5_NE",
        "uscrn-ms-newton-5-ene": "MS_Newton_5_ENE",
        "uscrn-ny-ithaca-13-e": "NY_Ithaca_13_E",
        "uscrn-ok-goodwell-2-e": "OK_Goodwell_2_E",
        "uscrn-ok-goodwell-2-se": "OK_Goodwell_2_SE",
    }
)
STATION_SENSORS = MappingProxyType({station: sensor for sensor, station in SENSOR_STATIONS.items()})

# Explicit observable allowlist. Labels and release metadata cannot become measurements.
MEASUREMENT_COLUMNS = (
    "SOIL_MOISTURE_5",
    "SOIL_MOISTURE_10",
    "P_CALC",
    "LATITUDE",
    "LONGITUDE",
    "T_HR_AVG",
    "T_MAX",
    "T_MIN",
    "SOLARAD",
    "SUR_TEMP",
    "RH_HR_AVG",
    "SOIL_MOISTURE_20",
    "SOIL_MOISTURE_50",
    "SOIL_MOISTURE_100",
    "SOIL_TEMP_5",
    "SOIL_TEMP_10",
    "SOIL_TEMP_20",
    "SOIL_TEMP_50",
    "SOIL_TEMP_100",
)
RELEASE_COLUMNS = frozenset(("station", "time", "partition", *MEASUREMENT_COLUMNS))
