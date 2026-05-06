# ============================================================
# EvapoGPT: AI-Assisted Evapotranspiration Analysis Dashboard
# ============================================================
#
# Developed by:
# Partha Pratim Ray, Sikkim University, May, 2026, parthapratimray1986@gmail.com
#
# Colab installation:
#   !pip install -q gradio openai pandas requests matplotlib
#
# Colab execution:
#   !python evapogpt_production_grade.py
#
# Purpose:
#   A Gradio-based research prototype for querying actual
#   evapotranspiration and FAO reference evapotranspiration ET0
#   for a valid place and date/range using Open-Meteo APIs and
#   OpenAI text generation.
#
# Production-grade revisions:
#   1. User prompt can be single-line or multi-line.
#   2. OpenAI first interprets whether the prompt is related to
#      evapotranspiration and extracts place/date/range.
#   3. If the prompt is not evapotranspiration-related, the app
#      gives a short OpenAI-generated redirection message and does
#      not call Open-Meteo.
#   4. Past dates/ranges are routed to the Open-Meteo Archive API.
#   5. Present/future dates/ranges are routed to the Open-Meteo
#      Forecast API.
#   6. Mixed past-to-future ranges are split safely into archive
#      and forecast segments, then merged.
#   7. Forecast requests are guarded against the Open-Meteo
#      forecast horizon of up to 16 days.
#   8. Missing/null Open-Meteo values are safely converted to NaN
#      and ignored during numeric aggregation.
#   9. A clean, light, publication-ready Gradio frontend is used.
# ============================================================


# ------------------------------------------------------------
# STEP 1: Import required libraries
# ------------------------------------------------------------

import os
import re
import json
import time
import tempfile
import datetime as dt
from typing import Dict, Tuple, Any, Optional, List

import requests
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from openai import OpenAI


# ------------------------------------------------------------
# STEP 2: Application configuration
# ------------------------------------------------------------

DEFAULT_OPENAI_MODEL_NAME = "gpt-5.4-mini"

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

APP_TITLE = "EvapoGPT: GPT API-Assisted Evapotranspiration Analysis Dashboard"
DEVELOPER_CREDIT = "Developed by Partha Pratim Ray, Sikkim University, May, 2026"

APP_DESCRIPTION = """
EvapoGPT retrieves location-specific actual evapotranspiration and FAO reference
evapotranspiration ET₀ data from Open-Meteo APIs and generates a structured
scientific interpretation using OpenAI text generation. The system supports
past, present, future, and date-range evapotranspiration queries.
"""

MAX_FORECAST_DAYS_AHEAD = 16
MAX_RANGE_DAYS = 31


# ------------------------------------------------------------
# STEP 3: Open-Meteo weather variables
# ------------------------------------------------------------

# Open-Meteo hourly variables used in the app.
# According to the Open-Meteo forecast API documentation:
# - evapotranspiration is a preceding-hour sum in mm.
# - et0_fao_evapotranspiration is FAO reference ET0, also a preceding-hour sum in mm.
# - vapour_pressure_deficit is instantaneous in kPa.
# - shortwave_radiation is a preceding-hour mean in W/m².
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "showers",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "vapour_pressure_deficit",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
]

# Daily variables available from the forecast API.
# Actual evapotranspiration is aggregated from hourly data.
DAILY_VARIABLES_FORECAST = [
    "temperature_2m_max",
    "temperature_2m_mean",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "showers_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

# Daily variables commonly supported by Open-Meteo archive API.
DAILY_VARIABLES_ARCHIVE = [
    "temperature_2m_max",
    "temperature_2m_mean",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

NUMERIC_WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "showers",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "vapour_pressure_deficit",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
]


# ------------------------------------------------------------
# STEP 4: Timing and date utilities
# ------------------------------------------------------------

def now_perf() -> float:
    """
    Returns high-resolution time in seconds.
    """
    return time.perf_counter()


def today_local() -> dt.date:
    """
    Returns the runtime-local date.
    In Colab/cloud environments, this usually follows the server timezone.
    The Open-Meteo request itself uses the resolved location timezone.
    """
    return dt.date.today()


def parse_iso_date_safely(date_text: Any) -> Optional[dt.date]:
    """
    Safely parses a date in YYYY-MM-DD format.
    """
    try:
        return dt.date.fromisoformat(str(date_text).strip())
    except Exception:
        return None


def iso_or_none(date_obj: Optional[dt.date]) -> Optional[str]:
    """
    Converts date object to ISO string if available.
    """
    return date_obj.isoformat() if date_obj else None


def clamp_date_range(start_date: dt.date, end_date: dt.date) -> Tuple[dt.date, dt.date]:
    """
    Ensures start_date <= end_date.
    """
    if end_date < start_date:
        return end_date, start_date
    return start_date, end_date


def range_length_days(start_date: dt.date, end_date: dt.date) -> int:
    """
    Inclusive range length in days.
    """
    return (end_date - start_date).days + 1


# ------------------------------------------------------------
# STEP 5: Fallback date/place extraction utilities
# ------------------------------------------------------------

def parse_relative_and_textual_dates(prompt_text: str) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    """
    Fallback local date parser used if OpenAI interpretation fails.

    Handles:
    - today
    - tomorrow
    - day after tomorrow
    - yesterday
    - YYYY-MM-DD
    - from YYYY-MM-DD to YYYY-MM-DD
    - between YYYY-MM-DD and YYYY-MM-DD
    - next N days
    - past/last N days
    """

    prompt = str(prompt_text).strip()
    lower_prompt = prompt.lower()
    today = today_local()

    # Date range: from 2026-05-01 to 2026-05-05
    range_match = re.search(
        r"\b(?:from|between)\s+(20\d{2}-\d{2}-\d{2})\s+(?:to|and|-)\s+(20\d{2}-\d{2}-\d{2})\b",
        lower_prompt
    )
    if range_match:
        d1 = parse_iso_date_safely(range_match.group(1))
        d2 = parse_iso_date_safely(range_match.group(2))
        if d1 and d2:
            return clamp_date_range(d1, d2)

    # All ISO dates
    iso_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", prompt)
    parsed_iso_dates = [parse_iso_date_safely(x) for x in iso_dates]
    parsed_iso_dates = [x for x in parsed_iso_dates if x is not None]

    if len(parsed_iso_dates) >= 2:
        return clamp_date_range(parsed_iso_dates[0], parsed_iso_dates[1])

    if len(parsed_iso_dates) == 1:
        return parsed_iso_dates[0], parsed_iso_dates[0]

    # Relative day expressions
    if "day after tomorrow" in lower_prompt:
        d = today + dt.timedelta(days=2)
        return d, d

    if "tomorrow" in lower_prompt:
        d = today + dt.timedelta(days=1)
        return d, d

    if "yesterday" in lower_prompt:
        d = today - dt.timedelta(days=1)
        return d, d

    if "today" in lower_prompt or "current" in lower_prompt or "present" in lower_prompt or "now" in lower_prompt:
        return today, today

    # next N days
    next_match = re.search(r"\bnext\s+(\d{1,2})\s+days?\b", lower_prompt)
    if next_match:
        n = max(1, int(next_match.group(1)))
        return today, today + dt.timedelta(days=n - 1)

    # past/last N days
    past_match = re.search(r"\b(?:past|last|previous)\s+(\d{1,2})\s+days?\b", lower_prompt)
    if past_match:
        n = max(1, int(past_match.group(1)))
        return today - dt.timedelta(days=n - 1), today

    return today, today


def extract_place_fallback(prompt_text: str) -> str:
    """
    Fallback location extractor used if OpenAI interpretation fails.
    """

    prompt = str(prompt_text).strip()

    place_patterns = [
        r"\bin\s+([A-Za-z\s\-,]+?)(?:\s+on|\s+from|\s+between|\s+for|\s+today|\s+tomorrow|\s+yesterday|\s+day after tomorrow|$)",
        r"\bat\s+([A-Za-z\s\-,]+?)(?:\s+on|\s+from|\s+between|\s+for|\s+today|\s+tomorrow|\s+yesterday|\s+day after tomorrow|$)",
        r"\bof\s+([A-Za-z\s\-,]+?)(?:\s+on|\s+from|\s+between|\s+for|\s+today|\s+tomorrow|\s+yesterday|\s+day after tomorrow|$)",
        r"\bfor\s+([A-Za-z\s\-,]+?)(?:\s+on|\s+from|\s+between|\s+today|\s+tomorrow|\s+yesterday|\s+day after tomorrow|$)",
    ]

    for pattern in place_patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)

        if match:
            candidate = match.group(1).strip(" ,.-")
            candidate = re.sub(
                r"\b(on|from|between|for|today|tomorrow|yesterday|day after tomorrow)\b.*$",
                "",
                candidate,
                flags=re.IGNORECASE
            ).strip(" ,.-")

            if len(candidate) >= 2:
                return candidate

    capitalized_phrases = re.findall(
        r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b",
        prompt
    )

    if capitalized_phrases:
        return capitalized_phrases[-1]

    return ""


def contains_evapotranspiration_semantics(prompt_text: str) -> bool:
    """
    Local fallback semantic gate for evapotranspiration-related prompts.
    """

    lower_prompt = str(prompt_text).lower()

    et_keywords = [
        "evapotranspiration",
        "evapo-transpiration",
        "evapo transpiration",
        "reference evapotranspiration",
        "actual evapotranspiration",
        "et0",
        "et₀",
        "eto",
        "fao",
        "penman",
        "monteith",
        "irrigation requirement",
        "crop water",
        "water loss",
        "plant transpiration",
        "soil evaporation",
        "vapour pressure deficit",
        "vapor pressure deficit",
        "vpd",
        "agricultural water",
        "hydrological",
    ]

    return any(keyword in lower_prompt for keyword in et_keywords)


# ------------------------------------------------------------
# STEP 6: OpenAI prompt interpretation
# ------------------------------------------------------------

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the first JSON object from a model response.
    """

    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Remove fenced code blocks if present.
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())

    try:
        return json.loads(text)
    except Exception:
        pass

    # Last attempt: find first {...}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def call_openai_text(api_key: str, model_name: str, system_instruction: str, input_text: str) -> Tuple[str, float]:
    """
    Calls the OpenAI Responses API and returns text output and delay.
    """

    if not api_key or not str(api_key).strip():
        raise ValueError("Please paste your OpenAI API key in the secure API key box.")

    if not model_name or not str(model_name).strip():
        model_name = DEFAULT_OPENAI_MODEL_NAME

    client = OpenAI(api_key=str(api_key).strip())

    start = now_perf()

    response = client.responses.create(
        model=str(model_name).strip(),
        instructions=system_instruction,
        input=input_text
    )

    delay = now_perf() - start

    return response.output_text, delay


def interpret_user_prompt_with_openai(
    api_key: str,
    model_name: str,
    user_prompt: str
) -> Tuple[Dict[str, Any], float]:
    """
    Uses OpenAI to determine:
    - whether the user is asking about evapotranspiration or related water-loss/ET0 concepts;
    - the target place;
    - start date and end date;
    - whether the request is a single-date or range request.

    The response is expected as strict JSON.
    """

    today = today_local().isoformat()

    system_instruction = f"""
You are a strict query interpretation engine for an evapotranspiration dashboard.

Current runtime date: {today}

Task:
Read the user's prompt and return a single valid JSON object only.

The dashboard can answer only evapotranspiration-related queries, including:
- actual evapotranspiration
- FAO reference evapotranspiration ET0 / ET₀ / ETo
- crop water demand related to ET or ET0
- irrigation-water interpretation based on ET or ET0
- soil evaporation + plant transpiration
- vapour/vapor pressure deficit only if connected to ET, crop water, or irrigation

If the prompt is not related to evapotranspiration or closely related agricultural/hydrological water-loss semantics,
return is_evapotranspiration_related=false.

Extract dates carefully:
- If user says today/current/present/now, use {today}.
- If user says tomorrow, use current date + 1 day.
- If user says yesterday, use current date - 1 day.
- If user says day after tomorrow, use current date + 2 days.
- If user gives one date, set start_date=end_date.
- If user gives a range, set start_date and end_date.
- If user gives next N days, start_date={today}, end_date=today+N-1.
- If user gives last/past/previous N days, end_date={today}, start_date=today-N+1.
- If no date is given but query is ET-related, use {today} for both start_date and end_date.

Location:
- Extract the most likely geographical place name.
- If no place can be inferred, set place to null.

Return exactly this JSON schema:
{{
  "is_evapotranspiration_related": true/false,
  "place": "string or null",
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "date_intent": "past|present|future|mixed|unknown",
  "query_type": "single_day|date_range|unknown",
  "evapotranspiration_terms_detected": ["list"],
  "reason": "brief reason"
}}

No markdown. No explanation outside JSON.
"""

    output_text, delay = call_openai_text(
        api_key=api_key,
        model_name=model_name,
        system_instruction=system_instruction,
        input_text=user_prompt
    )

    parsed = extract_json_object(output_text)

    if parsed is None:
        # Robust fallback if model does not return valid JSON.
        start_date, end_date = parse_relative_and_textual_dates(user_prompt)
        place = extract_place_fallback(user_prompt)
        related = contains_evapotranspiration_semantics(user_prompt)

        parsed = {
            "is_evapotranspiration_related": related,
            "place": place if place else None,
            "start_date": iso_or_none(start_date),
            "end_date": iso_or_none(end_date),
            "date_intent": "unknown",
            "query_type": "single_day" if start_date == end_date else "date_range",
            "evapotranspiration_terms_detected": [],
            "reason": "Fallback parser was used because JSON interpretation failed."
        }

    return parsed, delay


def normalize_interpretation(parsed: Dict[str, Any], user_prompt: str) -> Dict[str, Any]:
    """
    Normalizes OpenAI/fallback interpretation and fills safe defaults.
    """

    today = today_local()

    related = bool(parsed.get("is_evapotranspiration_related"))

    # If the LLM is too strict but obvious ET keywords exist, allow local fallback.
    if not related and contains_evapotranspiration_semantics(user_prompt):
        related = True

    place = parsed.get("place")
    if place is not None:
        place = str(place).strip()
        if place.lower() in ["null", "none", "unknown", ""]:
            place = None

    if related and not place:
        fallback_place = extract_place_fallback(user_prompt)
        place = fallback_place if fallback_place else None

    start_date = parse_iso_date_safely(parsed.get("start_date"))
    end_date = parse_iso_date_safely(parsed.get("end_date"))

    if related and (start_date is None or end_date is None):
        fallback_start, fallback_end = parse_relative_and_textual_dates(user_prompt)
        start_date = start_date or fallback_start or today
        end_date = end_date or fallback_end or start_date

    if start_date and end_date:
        start_date, end_date = clamp_date_range(start_date, end_date)

    query_type = "single_day"
    if start_date and end_date and start_date != end_date:
        query_type = "date_range"

    date_intent = "unknown"
    if start_date and end_date:
        if end_date < today:
            date_intent = "past"
        elif start_date > today:
            date_intent = "future"
        elif start_date == today and end_date == today:
            date_intent = "present"
        else:
            date_intent = "mixed"

    return {
        "is_evapotranspiration_related": related,
        "place": place,
        "start_date": iso_or_none(start_date),
        "end_date": iso_or_none(end_date),
        "date_intent": date_intent,
        "query_type": query_type,
        "evapotranspiration_terms_detected": parsed.get("evapotranspiration_terms_detected", []),
        "reason": parsed.get("reason", ""),
    }


def generate_non_evapotranspiration_reply(
    api_key: str,
    model_name: str,
    user_prompt: str
) -> Tuple[str, float]:
    """
    Generates a short reply when the user prompt is outside the scope.
    """

    system_instruction = """
You are EvapoGPT, an evapotranspiration data assistant.

The user asked something outside the scope of evapotranspiration, ET0, crop-water demand,
irrigation-water interpretation, or closely related environmental water-loss data.

Reply briefly and politely. Ask the user to ask only about evapotranspiration data,
including place and date/range. Give one example query.

Do not answer the unrelated question.
"""

    return call_openai_text(
        api_key=api_key,
        model_name=model_name,
        system_instruction=system_instruction,
        input_text=user_prompt
    )


# ------------------------------------------------------------
# STEP 7: Open-Meteo geocoding
# ------------------------------------------------------------

def geocode_location(place_name: str) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    Converts a place name into geographical metadata using Open-Meteo Geocoding API.
    """

    if not place_name or len(str(place_name).strip()) < 2:
        raise ValueError(
            "No valid location was detected. Please include a clear place name, "
            "for example: Gangtok, India or Berlin, Germany."
        )

    start = now_perf()

    params = {
        "name": str(place_name).strip(),
        "count": 10,
        "language": "en",
        "format": "json"
    }

    response = requests.get(GEOCODING_URL, params=params, timeout=30)
    delay = now_perf() - start

    response.raise_for_status()
    data = response.json()

    if data.get("error") is True:
        raise ValueError(f"Open-Meteo Geocoding error: {data.get('reason', 'Unknown reason')}")

    if "results" not in data or not data["results"]:
        raise ValueError(
            f"No geocoding result was found for location: {place_name}. "
            "Try a more specific place name, for example: 'Gangtok, India' or 'Berlin, Germany'."
        )

    result = data["results"][0]

    location_info = {
        "name": result.get("name"),
        "latitude": result.get("latitude"),
        "longitude": result.get("longitude"),
        "elevation": result.get("elevation"),
        "country": result.get("country"),
        "country_code": result.get("country_code"),
        "admin1": result.get("admin1"),
        "admin2": result.get("admin2"),
        "timezone": result.get("timezone", "auto"),
        "population": result.get("population"),
        "feature_code": result.get("feature_code"),
    }

    if location_info["latitude"] is None or location_info["longitude"] is None:
        raise ValueError(
            f"The geocoding result for {place_name} does not contain valid latitude/longitude."
        )

    return location_info, delay, data


# ------------------------------------------------------------
# STEP 8: Open-Meteo data fetching
# ------------------------------------------------------------

def clean_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans weather dataframe and safely converts numeric weather columns.
    """

    cleaned = df.copy()

    if "time" in cleaned.columns:
        cleaned["time"] = pd.to_datetime(cleaned["time"], errors="coerce")

    for col in NUMERIC_WEATHER_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned


def build_open_meteo_params(
    latitude: float,
    longitude: float,
    start_date: dt.date,
    end_date: dt.date,
    timezone: str,
    endpoint_kind: str
) -> Dict[str, Any]:
    """
    Builds Open-Meteo request parameters.
    """

    if endpoint_kind == "forecast":
        daily_vars = DAILY_VARIABLES_FORECAST
    else:
        daily_vars = DAILY_VARIABLES_ARCHIVE

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "daily": ",".join(daily_vars),
        "timezone": timezone or "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
        "cell_selection": "land",
    }

    return params


def fetch_segment_from_open_meteo(
    latitude: float,
    longitude: float,
    start_date: dt.date,
    end_date: dt.date,
    timezone: str,
    endpoint_kind: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], float, Dict[str, Any]]:
    """
    Fetches one segment from Open-Meteo.

    endpoint_kind:
    - "archive"
    - "forecast"
    """

    if endpoint_kind not in ["archive", "forecast"]:
        raise ValueError("Internal error: endpoint_kind must be 'archive' or 'forecast'.")

    endpoint_url = ARCHIVE_URL if endpoint_kind == "archive" else FORECAST_URL
    endpoint_label = "Open-Meteo Archive API" if endpoint_kind == "archive" else "Open-Meteo Forecast API"

    params = build_open_meteo_params(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
        endpoint_kind=endpoint_kind
    )

    start = now_perf()

    response = requests.get(endpoint_url, params=params, timeout=45)
    delay = now_perf() - start
    request_url = response.url

    # Open-Meteo returns JSON error objects for invalid parameters.
    try:
        data = response.json()
    except Exception:
        data = {}

    if response.status_code >= 400:
        reason = data.get("reason") if isinstance(data, dict) else response.text
        raise ValueError(f"{endpoint_label} error: {reason}")

    if isinstance(data, dict) and data.get("error") is True:
        raise ValueError(f"{endpoint_label} error: {data.get('reason', 'Unknown reason')}")

    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    if "time" not in hourly:
        raise ValueError(
            f"No hourly data was returned by {endpoint_label} for "
            f"{start_date.isoformat()} to {end_date.isoformat()}."
        )

    hourly_df = pd.DataFrame(hourly)
    daily_df = pd.DataFrame(daily) if daily else pd.DataFrame()

    if hourly_df.empty:
        raise ValueError(
            f"The returned hourly dataframe is empty for "
            f"{start_date.isoformat()} to {end_date.isoformat()}."
        )

    hourly_df = clean_weather_dataframe(hourly_df)

    if not daily_df.empty and "time" in daily_df.columns:
        daily_df["time"] = pd.to_datetime(daily_df["time"], errors="coerce")
        for col in daily_df.columns:
            if col != "time":
                daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")

    metadata = {
        "api_endpoint_used": endpoint_label,
        "endpoint_kind": endpoint_kind,
        "open_meteo_request_url": request_url,
        "segment_start_date": start_date.isoformat(),
        "segment_end_date": end_date.isoformat(),
        "latitude_returned": data.get("latitude"),
        "longitude_returned": data.get("longitude"),
        "elevation_returned": data.get("elevation"),
        "generationtime_ms": data.get("generationtime_ms"),
        "utc_offset_seconds": data.get("utc_offset_seconds"),
        "timezone": data.get("timezone"),
        "timezone_abbreviation": data.get("timezone_abbreviation"),
        "hourly_units": data.get("hourly_units", {}),
        "daily_units": data.get("daily_units", {}),
    }

    return hourly_df, daily_df, metadata, delay, data


def validate_requested_range(start_date: dt.date, end_date: dt.date) -> None:
    """
    Validates date range before calling APIs.
    """

    today = today_local()

    length = range_length_days(start_date, end_date)

    if length > MAX_RANGE_DAYS:
        raise ValueError(
            f"The requested date range contains {length} days. "
            f"For stable dashboard execution, please request at most {MAX_RANGE_DAYS} days at a time."
        )

    if end_date > today + dt.timedelta(days=MAX_FORECAST_DAYS_AHEAD - 1):
        raise ValueError(
            f"The requested range extends beyond Open-Meteo's practical forecast horizon. "
            f"Please keep future dates within the next {MAX_FORECAST_DAYS_AHEAD} days."
        )


def fetch_open_meteo_for_range(
    latitude: float,
    longitude: float,
    start_date_text: str,
    end_date_text: str,
    timezone: str = "auto"
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """
    Fetches Open-Meteo data for past, present, future, or mixed ranges.
    """

    start_date = parse_iso_date_safely(start_date_text)
    end_date = parse_iso_date_safely(end_date_text)

    if start_date is None or end_date is None:
        raise ValueError("Invalid date interpretation. Please use a clear date or date range.")

    start_date, end_date = clamp_date_range(start_date, end_date)
    validate_requested_range(start_date, end_date)

    today = today_local()

    segments: List[Tuple[dt.date, dt.date, str]] = []

    if end_date < today:
        segments.append((start_date, end_date, "archive"))

    elif start_date >= today:
        segments.append((start_date, end_date, "forecast"))

    else:
        # Mixed range: past part through yesterday from archive, today/future from forecast.
        archive_end = today - dt.timedelta(days=1)
        if start_date <= archive_end:
            segments.append((start_date, archive_end, "archive"))
        segments.append((today, end_date, "forecast"))

    hourly_frames = []
    daily_frames = []
    metadata_list = []
    raw_json_list = []
    total_delay = 0.0

    for seg_start, seg_end, endpoint_kind in segments:
        hourly_df, daily_df, metadata, delay, raw_json = fetch_segment_from_open_meteo(
            latitude=latitude,
            longitude=longitude,
            start_date=seg_start,
            end_date=seg_end,
            timezone=timezone,
            endpoint_kind=endpoint_kind
        )

        hourly_frames.append(hourly_df)
        if not daily_df.empty:
            daily_frames.append(daily_df)

        metadata_list.append(metadata)
        raw_json_list.append(raw_json)
        total_delay += delay

    combined_hourly = pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame()
    combined_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()

    if not combined_hourly.empty:
        combined_hourly = clean_weather_dataframe(combined_hourly)
        combined_hourly = combined_hourly.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    if not combined_daily.empty and "time" in combined_daily.columns:
        combined_daily = combined_daily.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    return combined_hourly, combined_daily, metadata_list, total_delay, raw_json_list


# ------------------------------------------------------------
# STEP 9: Summary metrics
# ------------------------------------------------------------

def safe_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """
    Returns a numeric series for a column after dropping missing values.
    """

    if df is None or df.empty or col not in df.columns:
        return None

    s = pd.to_numeric(df[col], errors="coerce").dropna()

    if s.empty:
        return None

    return s


def safe_float(value: Any) -> Optional[float]:
    """
    Converts a value to float if possible.
    """

    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def compute_overall_summary(hourly_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Computes overall summary metrics across the selected date/range.
    """

    summary: Dict[str, Optional[float]] = {}

    for col in NUMERIC_WEATHER_COLUMNS:
        s = safe_series(hourly_df, col)

        if s is None:
            summary[f"{col}_mean"] = None
            summary[f"{col}_min"] = None
            summary[f"{col}_max"] = None
        else:
            summary[f"{col}_mean"] = safe_float(s.mean())
            summary[f"{col}_min"] = safe_float(s.min())
            summary[f"{col}_max"] = safe_float(s.max())

    evap_s = safe_series(hourly_df, "evapotranspiration")
    et0_s = safe_series(hourly_df, "et0_fao_evapotranspiration")
    precip_s = safe_series(hourly_df, "precipitation")
    rain_s = safe_series(hourly_df, "rain")
    showers_s = safe_series(hourly_df, "showers")

    summary["actual_evapotranspiration_total_mm"] = safe_float(evap_s.sum()) if evap_s is not None else None
    summary["et0_fao_total_mm_from_hourly"] = safe_float(et0_s.sum()) if et0_s is not None else None
    summary["precipitation_total_mm"] = safe_float(precip_s.sum()) if precip_s is not None else None
    summary["rain_total_mm"] = safe_float(rain_s.sum()) if rain_s is not None else None
    summary["showers_total_mm"] = safe_float(showers_s.sum()) if showers_s is not None else None

    summary["temperature_mean_c"] = summary.get("temperature_2m_mean")
    summary["relative_humidity_mean_percent"] = summary.get("relative_humidity_2m_mean")
    summary["wind_speed_mean_kmh"] = summary.get("wind_speed_10m_mean")
    summary["vapour_pressure_deficit_mean_kpa"] = summary.get("vapour_pressure_deficit_mean")
    summary["shortwave_radiation_mean_wm2"] = summary.get("shortwave_radiation_mean")

    return summary


def compute_daily_summary_from_hourly(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates hourly values into daily summary rows.
    """

    if hourly_df is None or hourly_df.empty:
        return pd.DataFrame()

    df = clean_weather_dataframe(hourly_df)

    if "time" not in df.columns:
        return pd.DataFrame()

    df["date"] = df["time"].dt.date.astype(str)

    grouped_rows = []

    for date_text, group in df.groupby("date"):
        row = {"date": date_text}

        for col in NUMERIC_WEATHER_COLUMNS:
            s = safe_series(group, col)

            if s is not None:
                row[f"{col}_mean"] = safe_float(s.mean())
                row[f"{col}_min"] = safe_float(s.min())
                row[f"{col}_max"] = safe_float(s.max())
            else:
                row[f"{col}_mean"] = None
                row[f"{col}_min"] = None
                row[f"{col}_max"] = None

        evap_s = safe_series(group, "evapotranspiration")
        et0_s = safe_series(group, "et0_fao_evapotranspiration")
        precip_s = safe_series(group, "precipitation")

        row["actual_evapotranspiration_daily_sum_mm"] = safe_float(evap_s.sum()) if evap_s is not None else None
        row["et0_fao_daily_sum_mm_from_hourly"] = safe_float(et0_s.sum()) if et0_s is not None else None
        row["precipitation_daily_sum_mm"] = safe_float(precip_s.sum()) if precip_s is not None else None

        grouped_rows.append(row)

    daily_summary_df = pd.DataFrame(grouped_rows)

    return daily_summary_df


# ------------------------------------------------------------
# STEP 10: OpenAI scientific report generation
# ------------------------------------------------------------

def generate_openai_report(
    api_key: str,
    model_name: str,
    user_prompt: str,
    interpretation: Dict[str, Any],
    location_info: Dict[str, Any],
    metadata_list: List[Dict[str, Any]],
    overall_summary: Dict[str, Any],
    daily_summary_records: List[Dict[str, Any]],
    hourly_sample_records: List[Dict[str, Any]]
) -> Tuple[str, float]:
    """
    Generates a structured scientific report using OpenAI.
    """

    system_instruction = """
You are a scientific environmental-data interpretation assistant.

Generate a formal, publication-ready explanation of evapotranspiration data.

Strict rules:
1. Use only the supplied data.
2. Do not invent numerical values.
3. Explain actual evapotranspiration and FAO reference evapotranspiration ET0 separately.
4. Mention that evapotranspiration values are in millimetres.
5. Explain the role of temperature, humidity, wind speed, solar radiation,
   precipitation, and vapour pressure deficit when data are available.
6. Clearly state whether data came from forecast API, archive API, or both.
7. Mention if some values are missing, unavailable, interpolated, or model-derived.
8. Do not claim field-measured observations unless explicitly provided.
9. If the request covers multiple dates, summarize date-wise patterns without excessive repetition.
10. Keep the tone scientific, concise, and suitable for a research dashboard.

Required sections:
- Query Interpretation
- Location Metadata
- Date Range and API Source
- Key Numerical Findings
- Scientific Interpretation
- Agricultural or Environmental Implication
- Limitations and Cautions
- Concise Conclusion
"""

    payload = {
        "user_question": user_prompt,
        "interpreted_query": interpretation,
        "location_information": location_info,
        "open_meteo_metadata": metadata_list,
        "overall_summary_metrics": overall_summary,
        "daily_summary_records": daily_summary_records,
        "hourly_sample_records": hourly_sample_records,
    }

    input_payload = json.dumps(payload, indent=2, default=str)

    return call_openai_text(
        api_key=api_key,
        model_name=model_name,
        system_instruction=system_instruction,
        input_text=input_payload
    )


# ------------------------------------------------------------
# STEP 11: Chart generation
# ------------------------------------------------------------

def create_evapotranspiration_chart(
    hourly_df: pd.DataFrame,
    daily_summary_df: pd.DataFrame,
    place: str,
    start_date: str,
    end_date: str
) -> Optional[str]:
    """
    Creates a chart for evapotranspiration, ET0, and temperature.

    For single-day requests: hourly chart.
    For date-range requests: daily aggregate chart.
    """

    if hourly_df is None or hourly_df.empty:
        return None

    safe_place = re.sub(r"[^A-Za-z0-9_]+", "_", place or "location")

    is_single_day = start_date == end_date

    if is_single_day:
        plot_df = clean_weather_dataframe(hourly_df)

        fig, ax1 = plt.subplots(figsize=(13, 6.5))

        x = plot_df["time"]
        plotted_any = False

        if "evapotranspiration" in plot_df.columns and plot_df["evapotranspiration"].notna().any():
            ax1.bar(
                x,
                plot_df["evapotranspiration"],
                width=0.025,
                alpha=0.65,
                label="Actual Evapotranspiration (mm)"
            )
            plotted_any = True

        if "et0_fao_evapotranspiration" in plot_df.columns and plot_df["et0_fao_evapotranspiration"].notna().any():
            ax1.bar(
                x,
                plot_df["et0_fao_evapotranspiration"],
                width=0.018,
                alpha=0.45,
                label="FAO Reference ET₀ (mm)"
            )
            plotted_any = True

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Evapotranspiration / ET₀ (mm)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        ax2 = ax1.twinx()

        if "temperature_2m" in plot_df.columns and plot_df["temperature_2m"].notna().any():
            ax2.plot(
                x,
                plot_df["temperature_2m"],
                marker="o",
                linewidth=2.2,
                label="Temperature at 2 m (°C)"
            )
            plotted_any = True

        ax2.set_ylabel("Temperature (°C)")

        if not plotted_any:
            ax1.text(
                0.5,
                0.5,
                "No valid numeric values available for plotting.",
                transform=ax1.transAxes,
                ha="center",
                va="center",
                fontsize=13
            )

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        if labels_1 or labels_2:
            ax1.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc="upper left",
                frameon=True
            )

        plt.title(
            f"Hourly Evapotranspiration and Temperature: {place} on {start_date}",
            fontsize=14,
            fontweight="bold"
        )

        chart_name = f"evapotranspiration_hourly_{safe_place}_{start_date}.png"

    else:
        if daily_summary_df is None or daily_summary_df.empty:
            daily_summary_df = compute_daily_summary_from_hourly(hourly_df)

        plot_df = daily_summary_df.copy()
        plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")

        fig, ax1 = plt.subplots(figsize=(13, 6.5))

        x = plot_df["date"]
        plotted_any = False

        if "actual_evapotranspiration_daily_sum_mm" in plot_df.columns:
            y = pd.to_numeric(plot_df["actual_evapotranspiration_daily_sum_mm"], errors="coerce")
            if y.notna().any():
                ax1.bar(
                    x,
                    y,
                    alpha=0.65,
                    label="Daily Actual Evapotranspiration Sum (mm)"
                )
                plotted_any = True

        if "et0_fao_daily_sum_mm_from_hourly" in plot_df.columns:
            y2 = pd.to_numeric(plot_df["et0_fao_daily_sum_mm_from_hourly"], errors="coerce")
            if y2.notna().any():
                ax1.plot(
                    x,
                    y2,
                    marker="o",
                    linewidth=2.2,
                    label="Daily FAO Reference ET₀ Sum (mm)"
                )
                plotted_any = True

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Evapotranspiration / ET₀ (mm)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        ax2 = ax1.twinx()

        if "temperature_2m_mean" in plot_df.columns:
            temp = pd.to_numeric(plot_df["temperature_2m_mean"], errors="coerce")
            if temp.notna().any():
                ax2.plot(
                    x,
                    temp,
                    marker="s",
                    linewidth=2.0,
                    label="Daily Mean Temperature at 2 m (°C)"
                )
                plotted_any = True

        ax2.set_ylabel("Temperature (°C)")

        if not plotted_any:
            ax1.text(
                0.5,
                0.5,
                "No valid numeric values available for plotting.",
                transform=ax1.transAxes,
                ha="center",
                va="center",
                fontsize=13
            )

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        if labels_1 or labels_2:
            ax1.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc="upper left",
                frameon=True
            )

        plt.title(
            f"Daily Evapotranspiration and Temperature: {place}, {start_date} to {end_date}",
            fontsize=14,
            fontweight="bold"
        )

        chart_name = f"evapotranspiration_daily_{safe_place}_{start_date}_to_{end_date}.png"

    plt.tight_layout()

    chart_path = os.path.join(tempfile.gettempdir(), chart_name)
    fig.savefig(chart_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return chart_path


# ------------------------------------------------------------
# STEP 12: CSV saving with true cumulative append
# ------------------------------------------------------------

import shutil


def make_run_id() -> str:
    """
    Creates a unique run identifier.
    This helps distinguish rows generated from different dashboard executions.
    """

    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def prepare_dataframe_for_cumulative_save(
    df: pd.DataFrame,
    run_id: str,
    run_timestamp: str,
    place: str,
    start_date: str,
    end_date: str,
    dataset_type: str
) -> pd.DataFrame:
    """
    Adds stable metadata columns before cumulative CSV storage.
    """

    if df is None:
        df = pd.DataFrame()

    prepared = df.copy()

    # Even if a dataframe is empty, keep a proper schema for safer downloads.
    metadata = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "place": place,
        "query_start_date": start_date,
        "query_end_date": end_date,
        "dataset_type": dataset_type,
    }

    # Insert in reverse order so final order remains as specified.
    for col_name, col_value in reversed(metadata.items()):
        if col_name in prepared.columns:
            prepared[col_name] = col_value
        else:
            prepared.insert(0, col_name, col_value)

    return prepared


def append_or_create_cumulative_csv(new_df: pd.DataFrame, cumulative_csv_path: str) -> str:
    """
    Robust cumulative CSV writer.

    Important design:
    1. Read the existing cumulative CSV if it exists.
    2. Concatenate old rows and new rows in memory.
    3. Rewrite the full cumulative CSV.

    This avoids the common Gradio/Colab confusion where the downloaded file
    appears to contain only the latest run. The file returned for download is
    always created from the full cumulative table.
    """

    os.makedirs(os.path.dirname(cumulative_csv_path), exist_ok=True)

    if new_df is None:
        new_df = pd.DataFrame()

    new_df = new_df.copy()

    if os.path.exists(cumulative_csv_path) and os.path.getsize(cumulative_csv_path) > 0:
        try:
            old_df = pd.read_csv(cumulative_csv_path, encoding="utf-8-sig")
        except Exception:
            # Fallback for files written without BOM or with default UTF-8.
            old_df = pd.read_csv(cumulative_csv_path)

        # Align columns safely. This is important because daily API columns may
        # vary between archive and forecast API calls.
        cumulative_df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        cumulative_df = new_df

    cumulative_df.to_csv(
        cumulative_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    return cumulative_csv_path


def make_download_snapshot(cumulative_csv_path: str, run_id: str, label: str) -> str:
    """
    Creates a fresh downloadable copy of the cumulative CSV for Gradio.

    Returning a fresh file path avoids browser/Gradio caching and ensures that
    the user downloads the full updated cumulative file after every run.
    """

    snapshot_dir = os.path.join(tempfile.gettempdir(), "evapogpt_download_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    snapshot_path = os.path.join(
        snapshot_dir,
        f"{label}_cumulative_download_{run_id}.csv"
    )

    shutil.copyfile(cumulative_csv_path, snapshot_path)

    return snapshot_path


def save_csv_files(
    hourly_df: pd.DataFrame,
    daily_summary_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    place: str,
    start_date: str,
    end_date: str
) -> Tuple[str, str, str]:
    """
    Saves and returns cumulative CSV files for:
    1. Hourly Raw Data
    2. Daily Summary Data
    3. Evaluation and Latency Metrics

    Key correction:
    The returned downloadable files are cumulative across all successful runs,
    not merely the latest run. Therefore, if the current query produces 3 daily
    rows, the downloaded Daily Summary CSV will contain earlier rows plus these
    3 new rows.
    """

    # Persistent cumulative output folder.
    # In Google Colab this will remain available during the active runtime.
    csv_dir = os.path.join(os.getcwd(), "evapogpt_csv_outputs")
    os.makedirs(csv_dir, exist_ok=True)

    # Use global cumulative files, not date-specific files.
    # This is essential for true append behavior across different dates/places.
    hourly_cumulative_path = os.path.join(
        csv_dir,
        "hourly_raw_data_all_queries_cumulative.csv"
    )

    daily_cumulative_path = os.path.join(
        csv_dir,
        "daily_summary_all_queries_cumulative.csv"
    )

    evaluation_cumulative_path = os.path.join(
        csv_dir,
        "evaluation_latency_metrics_all_queries_cumulative.csv"
    )

    run_id = make_run_id()
    run_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hourly_to_save = prepare_dataframe_for_cumulative_save(
        df=hourly_df,
        run_id=run_id,
        run_timestamp=run_timestamp,
        place=place,
        start_date=start_date,
        end_date=end_date,
        dataset_type="hourly_raw_data"
    )

    daily_to_save = prepare_dataframe_for_cumulative_save(
        df=daily_summary_df,
        run_id=run_id,
        run_timestamp=run_timestamp,
        place=place,
        start_date=start_date,
        end_date=end_date,
        dataset_type="daily_summary"
    )

    evaluation_to_save = prepare_dataframe_for_cumulative_save(
        df=evaluation_df,
        run_id=run_id,
        run_timestamp=run_timestamp,
        place=place,
        start_date=start_date,
        end_date=end_date,
        dataset_type="evaluation_latency_metrics"
    )

    append_or_create_cumulative_csv(hourly_to_save, hourly_cumulative_path)
    append_or_create_cumulative_csv(daily_to_save, daily_cumulative_path)
    append_or_create_cumulative_csv(evaluation_to_save, evaluation_cumulative_path)

    # Return fresh downloadable snapshots of the cumulative master files.
    hourly_download_path = make_download_snapshot(
        hourly_cumulative_path,
        run_id,
        "hourly_raw_data_all_queries"
    )

    daily_download_path = make_download_snapshot(
        daily_cumulative_path,
        run_id,
        "daily_summary_all_queries"
    )

    evaluation_download_path = make_download_snapshot(
        evaluation_cumulative_path,
        run_id,
        "evaluation_latency_metrics_all_queries"
    )

    return hourly_download_path, daily_download_path, evaluation_download_path


# ------------------------------------------------------------
# STEP 13: Main EvapoGPT pipeline
# ------------------------------------------------------------

def run_evapogpt(api_key: str, model_name: str, user_prompt: str):
    """
    Main Gradio callback function.
    """

    total_start = now_perf()

    try:
        # ----------------------------------------------------
        # 13.1: Basic input checks
        # ----------------------------------------------------

        if not user_prompt or len(str(user_prompt).strip()) < 3:
            raise ValueError(
                "Please enter a meaningful query, for example: "
                "What is the evapotranspiration level in Gangtok, India today?"
            )

        if not api_key or not str(api_key).strip():
            raise ValueError("Please paste your OpenAI API key in the secure API key box.")

        if not model_name or not str(model_name).strip():
            model_name = DEFAULT_OPENAI_MODEL_NAME

        # ----------------------------------------------------
        # 13.2: Interpret prompt with OpenAI
        # ----------------------------------------------------

        raw_interpretation, interpretation_delay = interpret_user_prompt_with_openai(
            api_key=api_key,
            model_name=model_name,
            user_prompt=user_prompt
        )

        interpretation = normalize_interpretation(raw_interpretation, user_prompt)

        # ----------------------------------------------------
        # 13.3: If not ET-related, generate scope reply only
        # ----------------------------------------------------

        if not interpretation.get("is_evapotranspiration_related"):
            scope_reply, scope_delay = generate_non_evapotranspiration_reply(
                api_key=api_key,
                model_name=model_name,
                user_prompt=user_prompt
            )

            total_delay = now_perf() - total_start

            report_output = f"""
# EvapoGPT Scope Notice

{scope_reply}

---

**System note:** No Open-Meteo weather request was made because the prompt was not interpreted as evapotranspiration-related.

**OpenAI interpretation delay:** {interpretation_delay:.4f} seconds
**OpenAI scope-reply delay:** {scope_delay:.4f} seconds
**Total delay:** {total_delay:.4f} seconds

---

**{DEVELOPER_CREDIT}**
"""

            empty_df = pd.DataFrame(
                {
                    "message": [
                        "Prompt was outside evapotranspiration scope. No Open-Meteo data was requested."
                    ]
                }
            )

            return (
                report_output,
                None,
                empty_df,
                empty_df,
                None,
                None,
                None
            )

        # ----------------------------------------------------
        # 13.4: Validate interpreted ET request
        # ----------------------------------------------------

        place = interpretation.get("place")
        start_date_text = interpretation.get("start_date")
        end_date_text = interpretation.get("end_date")

        if not place:
            raise ValueError(
                "The prompt appears to be about evapotranspiration, but no valid place was detected. "
                "Please include a place name, for example: Gangtok, India."
            )

        start_date = parse_iso_date_safely(start_date_text)
        end_date = parse_iso_date_safely(end_date_text)

        if start_date is None or end_date is None:
            raise ValueError(
                "The prompt appears to be about evapotranspiration, but no valid date or date range was detected. "
                "Please use a clear date such as today, tomorrow, 2026-05-10, or from 2026-05-01 to 2026-05-05."
            )

        start_date, end_date = clamp_date_range(start_date, end_date)
        validate_requested_range(start_date, end_date)

        interpretation["start_date"] = start_date.isoformat()
        interpretation["end_date"] = end_date.isoformat()
        interpretation["query_type"] = "single_day" if start_date == end_date else "date_range"

        # ----------------------------------------------------
        # 13.5: Geocode location
        # ----------------------------------------------------

        location_info, geocoding_delay, raw_geocoding_json = geocode_location(place)

        # ----------------------------------------------------
        # 13.6: Fetch Open-Meteo data according to date scope
        # ----------------------------------------------------

        hourly_df, daily_api_df, metadata_list, open_meteo_delay, raw_json_list = fetch_open_meteo_for_range(
            latitude=location_info["latitude"],
            longitude=location_info["longitude"],
            start_date_text=start_date.isoformat(),
            end_date_text=end_date.isoformat(),
            timezone=location_info.get("timezone", "auto")
        )

        # ----------------------------------------------------
        # 13.7: Local processing and summaries
        # ----------------------------------------------------

        local_processing_start = now_perf()

        overall_summary = compute_overall_summary(hourly_df)
        daily_summary_df = compute_daily_summary_from_hourly(hourly_df)

        # Merge daily API ET0 values if available, but keep hourly-derived daily summary primary.
        if not daily_api_df.empty:
            daily_api_copy = daily_api_df.copy()
            if "time" in daily_api_copy.columns:
                daily_api_copy["date"] = daily_api_copy["time"].dt.date.astype(str)
                merge_cols = ["date"]
                for col in daily_api_copy.columns:
                    if col not in ["time", "date"]:
                        merge_cols.append(col)

                daily_summary_df = daily_summary_df.merge(
                    daily_api_copy[merge_cols],
                    on="date",
                    how="left",
                    suffixes=("", "_daily_api")
                )

        local_processing_delay = now_perf() - local_processing_start

        # ----------------------------------------------------
        # 13.8: Generate chart
        # ----------------------------------------------------

        chart_start = now_perf()

        chart_path = create_evapotranspiration_chart(
            hourly_df=hourly_df,
            daily_summary_df=daily_summary_df,
            place=place,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )

        chart_delay = now_perf() - chart_start

        # ----------------------------------------------------
        # 13.9: Generate OpenAI scientific report
        # ----------------------------------------------------

        hourly_sample_records = hourly_df.head(48).to_dict(orient="records")
        daily_summary_records = daily_summary_df.to_dict(orient="records")

        report, openai_report_delay = generate_openai_report(
            api_key=api_key,
            model_name=model_name,
            user_prompt=user_prompt,
            interpretation=interpretation,
            location_info=location_info,
            metadata_list=metadata_list,
            overall_summary=overall_summary,
            daily_summary_records=daily_summary_records,
            hourly_sample_records=hourly_sample_records
        )

        # ----------------------------------------------------
        # 13.10: Evaluation and latency table
        # ----------------------------------------------------

        total_delay = now_perf() - total_start
        network_delay = geocoding_delay + open_meteo_delay + openai_report_delay

        endpoint_names = "; ".join([m.get("api_endpoint_used", "") for m in metadata_list])
        request_urls = " | ".join([m.get("open_meteo_request_url", "") for m in metadata_list])

        evaluation_data = {
            "query": [user_prompt],
            "is_evapotranspiration_related": [interpretation.get("is_evapotranspiration_related")],
            "parsed_place": [place],
            "start_date": [start_date.isoformat()],
            "end_date": [end_date.isoformat()],
            "query_type": [interpretation.get("query_type")],
            "date_intent": [interpretation.get("date_intent")],
            "country": [location_info.get("country")],
            "country_code": [location_info.get("country_code")],
            "admin1": [location_info.get("admin1")],
            "admin2": [location_info.get("admin2")],
            "latitude": [location_info.get("latitude")],
            "longitude": [location_info.get("longitude")],
            "elevation_m": [location_info.get("elevation")],
            "timezone": [location_info.get("timezone")],
            "open_meteo_endpoint_used": [endpoint_names],
            "open_meteo_request_url": [request_urls],
            "actual_evapotranspiration_total_mm": [
                overall_summary.get("actual_evapotranspiration_total_mm")
            ],
            "et0_fao_total_mm_from_hourly": [
                overall_summary.get("et0_fao_total_mm_from_hourly")
            ],
            "precipitation_total_mm": [
                overall_summary.get("precipitation_total_mm")
            ],
            "temperature_mean_c": [
                overall_summary.get("temperature_mean_c")
            ],
            "relative_humidity_mean_percent": [
                overall_summary.get("relative_humidity_mean_percent")
            ],
            "wind_speed_mean_kmh": [
                overall_summary.get("wind_speed_mean_kmh")
            ],
            "vapour_pressure_deficit_mean_kpa": [
                overall_summary.get("vapour_pressure_deficit_mean_kpa")
            ],
            "shortwave_radiation_mean_wm2": [
                overall_summary.get("shortwave_radiation_mean_wm2")
            ],
            "openai_interpretation_delay_sec": [interpretation_delay],
            "geocoding_api_delay_sec": [geocoding_delay],
            "open_meteo_weather_api_delay_sec": [open_meteo_delay],
            "local_processing_delay_sec": [local_processing_delay],
            "chart_generation_delay_sec": [chart_delay],
            "openai_report_generation_delay_sec": [openai_report_delay],
            "network_delay_sec": [network_delay],
            "total_delay_sec": [total_delay],
            "model_used": [model_name],
            "interpretation_reason": [interpretation.get("reason")],
        }

        evaluation_df = pd.DataFrame(evaluation_data)

        # ----------------------------------------------------
        # 13.11: Save CSV files
        # ----------------------------------------------------

        hourly_csv_path, daily_csv_path, evaluation_csv_path = save_csv_files(
            hourly_df=hourly_df,
            daily_summary_df=daily_summary_df,
            evaluation_df=evaluation_df,
            place=place,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )

        # ----------------------------------------------------
        # 13.12: Prepare display tables
        # ----------------------------------------------------

        display_daily_df = daily_summary_df.copy()
        display_eval_df = evaluation_df.copy()

        for frame in [display_daily_df, display_eval_df]:
            for col in frame.columns:
                if pd.api.types.is_numeric_dtype(frame[col]):
                    frame[col] = frame[col].round(6)

        # ----------------------------------------------------
        # 13.13: Markdown report
        # ----------------------------------------------------

        report_output = f"""
# EvapoGPT Analysis Report

{report}

---

## Technical Note

The values are model-derived weather and evapotranspiration estimates obtained through Open-Meteo APIs.
Actual evapotranspiration and FAO reference evapotranspiration ET₀ are reported in millimetres when available.
Hourly actual evapotranspiration is aggregated locally to produce daily or range-level values.
Some API variables may contain missing values for some places, dates, or weather models; the app safely ignores missing
numeric values when computing sums and averages. For irrigation planning, agricultural decision-making,
hydrological modelling, or environmental field studies, these values should be validated with local field measurements whenever possible.

---

**{DEVELOPER_CREDIT}**
"""

        return (
            report_output,
            chart_path,
            display_daily_df,
            display_eval_df,
            hourly_csv_path,
            daily_csv_path,
            evaluation_csv_path
        )

    except Exception as e:
        error_report = f"""
# EvapoGPT Error Report

The app could not complete the requested analysis.

## Error Message

```text
{str(e)}
```

## Suggested Fixes

1. Ask specifically about evapotranspiration, ET₀, irrigation water demand, crop water loss, or related environmental water-loss data.
2. Include a clear place name, for example: `Gangtok, India`, `Berlin, Germany`, or `Delhi, India`.
3. Include a clear date or date range, for example: `today`, `tomorrow`, `2026-05-10`, or `from 2026-05-01 to 2026-05-05`.
4. Keep future dates within Open-Meteo's supported forecast horizon of about 16 days.
5. Keep date ranges within {MAX_RANGE_DAYS} days for stable dashboard execution.
6. Check whether your OpenAI API key is valid and whether your account has access to the selected model.
7. If the selected OpenAI model is unavailable, try another model enabled in your account.

---

**{DEVELOPER_CREDIT}**
"""

        empty_df = pd.DataFrame({"error": [str(e)]})

        return (
            error_report,
            None,
            empty_df,
            empty_df,
            None,
            None,
            None
        )


# ------------------------------------------------------------
# STEP 14: Professional light frontend CSS
# ------------------------------------------------------------

custom_css = """
/* ============================================================
   EvapoGPT Professional Light Theme
   Clean publication-ready frontend with clear text visibility
   ============================================================ */

.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
    background: #f8fafc !important;
    color: #111827 !important;
    font-family: "Inter", "Segoe UI", Arial, sans-serif !important;
}

.gradio-container,
.gradio-container * {
    color: #111827 !important;
}

.block,
.form,
.wrap,
.panel,
.tabs,
.tabitem {
    background: #ffffff !important;
    color: #111827 !important;
}

#title-card {
    background: linear-gradient(135deg, #ffffff 0%, #eef6ff 52%, #e8f3ff 100%) !important;
    border-radius: 24px;
    padding: 36px 32px;
    margin: 18px 0 22px 0;
    color: #0f172a !important;
    box-shadow: 0 12px 28px rgba(30, 64, 175, 0.12);
    border: 1px solid #bfdbfe;
}

#title-card h1 {
    font-size: 36px;
    font-weight: 850;
    line-height: 1.2;
    margin: 0 0 12px 0;
    color: #0f172a !important;
    letter-spacing: 0.2px;
}

#title-card p {
    font-size: 16px;
    line-height: 1.7;
    color: #1f2937 !important;
    margin: 0;
}

#developer-credit {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid #c7d2fe;
    font-size: 15px;
    font-weight: 750;
    color: #1e3a8a !important;
}

.info-card {
    background: #ffffff !important;
    border-radius: 18px;
    border: 1px solid #dbeafe;
    padding: 20px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
    color: #111827 !important;
}

.prose,
.markdown,
.md,
.markdown-body,
.prose *,
.markdown *,
.md *,
.markdown-body * {
    background: transparent !important;
    color: #111827 !important;
}

.prose h1,
.prose h2,
.prose h3,
.prose h4,
.markdown h1,
.markdown h2,
.markdown h3,
.markdown h4,
.md h1,
.md h2,
.md h3,
.md h4 {
    color: #0f172a !important;
    font-weight: 800 !important;
    background: transparent !important;
}

.prose p,
.prose li,
.prose span,
.prose strong,
.markdown p,
.markdown li,
.markdown span,
.markdown strong,
.md p,
.md li,
.md span,
.md strong {
    color: #111827 !important;
    background: transparent !important;
}

blockquote {
    background: transparent !important;
    border-left: 4px solid #2563eb !important;
    color: #111827 !important;
    padding-left: 14px !important;
    margin-left: 0 !important;
}

pre,
code {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    border-radius: 8px !important;
}

label {
    font-weight: 750 !important;
    color: #111827 !important;
    background: transparent !important;
}

textarea,
input,
.input,
.wrap textarea,
.wrap input {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
}

textarea::placeholder,
input::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

button {
    border-radius: 12px !important;
    font-weight: 750 !important;
}

button.primary,
.primary {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #1d4ed8 !important;
}

.tab-nav,
.tab-nav button,
.tabs button {
    background: #ffffff !important;
    color: #111827 !important;
    font-weight: 750 !important;
}

.selected,
button.selected {
    background: #e0f2fe !important;
    color: #0f172a !important;
    border-color: #38bdf8 !important;
}

.report-box {
    background: #ffffff !important;
    border-radius: 18px !important;
    border: 1px solid #dbeafe !important;
    padding: 22px !important;
    color: #111827 !important;
}

.report-box *,
.report-box .markdown,
.report-box .prose,
.report-box p,
.report-box li,
.report-box h1,
.report-box h2,
.report-box h3,
.report-box h4,
.report-box span,
.report-box strong {
    background: transparent !important;
}

table,
th,
td,
.dataframe,
.table-wrap {
    background: #ffffff !important;
    color: #111827 !important;
    border-color: #e5e7eb !important;
}

th {
    background: #eff6ff !important;
    color: #0f172a !important;
    font-weight: 800 !important;
}

td {
    background: #ffffff !important;
    color: #111827 !important;
}

.table-wrap,
.dataframe,
tbody,
thead {
    color: #111827 !important;
}

.file-preview,
.file-preview *,
.file-preview a,
.file-preview span,
.file-preview div,
.file-preview p {
    background: transparent !important;
    color: #111827 !important;
}

.file-preview {
    border: 1px solid #dbeafe !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

[data-testid="file"],
[data-testid="file"] *,
.download,
.download *,
.file,
.file * {
    background: transparent !important;
    color: #111827 !important;
}

#footer-note {
    text-align: center;
    color: #334155 !important;
    font-size: 14px;
    font-weight: 600;
    padding: 18px;
    background: transparent !important;
}
"""


# ------------------------------------------------------------
# STEP 15: Build Gradio interface
# ------------------------------------------------------------

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML(
        f"""
        <div id="title-card">
            <h1>{APP_TITLE}</h1>
            <p>{APP_DESCRIPTION}</p>
            <div id="developer-credit">{DEVELOPER_CREDIT}</div>
        </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            with gr.Group(elem_classes=["info-card"]):

                gr.Markdown("## Input Panel")

                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="Paste your OpenAI API key here. It is used only during this runtime session.",
                    type="password",
                    lines=1
                )

                model_name_input = gr.Textbox(
                    label="OpenAI Model Name",
                    value=DEFAULT_OPENAI_MODEL_NAME,
                    placeholder="Example: gpt-5.4-mini"
                )

                prompt_input = gr.Textbox(
                    label="Ask about evapotranspiration",
                    placeholder=(
                        "Examples:\n"
                        "1. What is the evapotranspiration level in Gangtok, India today?\n"
                        "2. Give ET0 for Berlin from 2026-05-06 to 2026-05-10.\n"
                        "3. Analyze actual evapotranspiration and irrigation implications for Delhi tomorrow."
                    ),
                    lines=7
                )

                run_button = gr.Button(
                    value="Generate Evapotranspiration Report",
                    variant="primary"
                )

        with gr.Column(scale=1):

            with gr.Group(elem_classes=["info-card"]):

                gr.Markdown(
                    """
                    ## Usage Guidance

                    **The prompt must be related to evapotranspiration.**

                    The app understands prompts about:

                    - actual evapotranspiration
                    - FAO reference evapotranspiration ET₀ / ET0 / ETo
                    - crop water demand related to ET or ET₀
                    - irrigation interpretation based on ET or ET₀
                    - soil evaporation and plant transpiration
                    - vapour pressure deficit when related to evapotranspiration

                    **Recommended examples**

                    `What is the evapotranspiration level in Gangtok, India today?`

                    `Give ET0 for Berlin, Germany from 2026-05-06 to 2026-05-10.`

                    `Analyze actual evapotranspiration and irrigation implication for Delhi tomorrow.`

                    **Date support**

                    - `today`
                    - `tomorrow`
                    - `yesterday`
                    - `day after tomorrow`
                    - `YYYY-MM-DD`
                    - `from YYYY-MM-DD to YYYY-MM-DD`
                    - `next 5 days`
                    - `past 7 days`

                    **API routing**

                    - Past dates are routed to the Open-Meteo Archive API.
                    - Present and future dates are routed to the Open-Meteo Forecast API.
                    - Mixed past-to-future ranges are split and merged automatically.
                    - Future dates must stay within Open-Meteo's forecast horizon.
                    """
                )

    with gr.Tab("GPT-Generated Report"):

        report_output = gr.Markdown(
            label="Generated Report",
            elem_classes=["report-box"]
        )

    with gr.Tab("Chart"):

        chart_output = gr.Image(
            label="Evapotranspiration and Temperature Chart",
            type="filepath"
        )

    with gr.Tab("Daily Summary Data"):

        daily_table_output = gr.Dataframe(
            label="Daily Evapotranspiration Summary",
            interactive=False,
            wrap=True
        )

        daily_csv_output = gr.File(
            label="Download Daily Summary CSV"
        )

    with gr.Tab("Evaluation and Latency Metrics"):

        evaluation_table_output = gr.Dataframe(
            label="Evaluation Metrics and Delay Analysis",
            interactive=False,
            wrap=True
        )

        evaluation_csv_output = gr.File(
            label="Download Evaluation Metrics CSV"
        )

    with gr.Tab("Hourly Raw Data CSV"):

        hourly_csv_output = gr.File(
            label="Download Hourly Open-Meteo Data CSV"
        )

    gr.HTML(
        f"""
        <div id="footer-note">
            {DEVELOPER_CREDIT}
        </div>
        """
    )

    run_button.click(
        fn=run_evapogpt,
        inputs=[
            api_key_input,
            model_name_input,
            prompt_input
        ],
        outputs=[
            report_output,
            chart_output,
            daily_table_output,
            evaluation_table_output,
            hourly_csv_output,
            daily_csv_output,
            evaluation_csv_output
        ]
    )


# ------------------------------------------------------------
# STEP 16: Launch app
# ------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
