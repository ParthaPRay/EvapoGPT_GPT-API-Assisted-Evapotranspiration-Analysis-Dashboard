# EvapoGPT: GPT API-Assisted Evapotranspiration Analysis Dashboard

**EvapoGPT** is a Gradio-based research prototype for retrieving, analyzing, visualizing, and interpreting evapotranspiration-related weather data using the **Open-Meteo APIs** and the **OpenAI GPT API**.

The system allows users to ask natural-language questions about **actual evapotranspiration**, **FAO reference evapotranspiration ET₀**, crop-water demand, irrigation implications, and related environmental water-loss conditions for a valid location and date or date range.

---

## Developed By

**Partha Pratim Ray**  
Department of Computer Applications  
Sikkim University, India  
May, 2026

---

## Project Overview

Evapotranspiration is a key hydrological and agricultural variable that combines water loss from soil evaporation and plant transpiration. It is important for irrigation planning, crop-water demand estimation, agro-climatic analysis, and environmental monitoring.

EvapoGPT integrates:

- natural-language query interpretation,
- geocoding of user-specified locations,
- Open-Meteo weather and evapotranspiration data retrieval,
- hourly and daily evapotranspiration summarization,
- latency and evaluation logging,
- cumulative CSV storage,
- scientific GPT-generated interpretation,
- publication-ready visualization,
- and a clean Gradio web interface.

---

## Main Features

### 1. Natural-Language Evapotranspiration Querying

Users can ask simple or complex prompts such as:

```text
What is the evapotranspiration level in Gangtok, India today?
````

```text
Give ET0 for Ludhiana, India from 2026-05-06 to 2026-05-10.
```

```text
Analyze actual evapotranspiration and irrigation implications for Coimbatore, India tomorrow.
```

The prompt may be either single-line or multi-line.

---

### 2. GPT-Based Query Interpretation

The system uses the OpenAI GPT API to identify whether the query is related to evapotranspiration. It extracts:

* location,
* start date,
* end date,
* date intent,
* query type,
* evapotranspiration-related terms,
* and interpretation reason.

If the prompt is not related to evapotranspiration, the system does not call Open-Meteo and instead returns a polite scope notice.

---

### 3. Open-Meteo API Integration

EvapoGPT uses the following Open-Meteo services:

| API                      | Purpose                                                                        |
| ------------------------ | ------------------------------------------------------------------------------ |
| Open-Meteo Geocoding API | Converts place names into latitude, longitude, timezone, and location metadata |
| Open-Meteo Forecast API  | Retrieves present and future evapotranspiration/weather data                   |
| Open-Meteo Archive API   | Retrieves past evapotranspiration/weather data                                 |

---

### 4. Automatic Date-Aware API Routing

The system automatically routes requests according to date type:

| Date Type                  | API Used                                              |
| -------------------------- | ----------------------------------------------------- |
| Past dates                 | Open-Meteo Archive API                                |
| Present date               | Open-Meteo Forecast API                               |
| Future dates               | Open-Meteo Forecast API                               |
| Mixed past-to-future range | Split into archive and forecast segments, then merged |

---

### 5. Supported Date Expressions

EvapoGPT supports common date expressions, including:

```text
today
tomorrow
yesterday
day after tomorrow
2026-05-10
from 2026-05-01 to 2026-05-05
next 5 days
past 7 days
```

The app restricts very long requests to maintain stable dashboard performance.

---

### 6. Weather and Evapotranspiration Variables

The system retrieves hourly weather and evapotranspiration variables such as:

* temperature at 2 m,
* relative humidity,
* precipitation,
* rain,
* showers,
* wind speed,
* wind gusts,
* shortwave radiation,
* vapour pressure deficit,
* actual evapotranspiration,
* FAO reference evapotranspiration ET₀.

---

### 7. Scientific Report Generation

After retrieving data, EvapoGPT generates a structured GPT-assisted report containing:

* Query Interpretation,
* Location Metadata,
* Date Range and API Source,
* Key Numerical Findings,
* Scientific Interpretation,
* Agricultural or Environmental Implication,
* Limitations and Cautions,
* Concise Conclusion.

The report is generated strictly from the retrieved data and does not invent numerical values.

---

### 8. Visualization

The system generates a chart for evapotranspiration and temperature.

For single-day queries, it produces an hourly plot.

For date-range queries, it produces a daily aggregate plot showing:

* daily actual evapotranspiration,
* daily FAO reference ET₀,
* and daily mean temperature where available.

---

### 9. Cumulative CSV Output

EvapoGPT saves and returns three cumulative CSV outputs:

| CSV File                           | Description                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------- |
| Hourly Raw Data CSV                | Hourly Open-Meteo data for all successful queries                                  |
| Daily Summary CSV                  | Daily aggregated evapotranspiration and weather summaries                          |
| Evaluation and Latency Metrics CSV | Query metadata, API delays, processing delays, model used, and key summary metrics |

The CSV-saving logic is designed to append new results to previous records rather than overwriting the latest output only.

---

## Repository Structure

A typical repository may be organized as follows:

```text
EvapoGPT/
│
├── evapogpt_production_grade.py
├── README.md
├── requirements.txt
├── outputs/
│   ├── charts/
│   └── csv/
└── screenshots/
```

The main executable file is:

```text
evapogpt_production_grade.py
```

---

## Installation

### Option 1: Local Installation

Clone the repository:

```bash
git clone https://github.com/your-username/EvapoGPT.git
cd EvapoGPT
```

Install dependencies:

```bash
pip install gradio openai pandas requests matplotlib
```

Run the application:

```bash
python evapogpt_production_grade.py
```

---

### Option 2: Google Colab Installation

Install the required packages:

```python
!pip install -q gradio openai pandas requests matplotlib
```

Run the script:

```python
!python evapogpt_production_grade.py
```

When running in Colab, the app launches a Gradio interface. If `share=True` is enabled, a temporary public URL is generated.

---

## Requirements

Create a `requirements.txt` file with:

```text
gradio
openai
pandas
requests
matplotlib
```

Install using:

```bash
pip install -r requirements.txt
```

---

## API Key Requirement

EvapoGPT requires an **OpenAI API key** to perform:

* natural-language prompt interpretation,
* non-scope redirection message generation,
* and scientific report generation.

The key is entered through the Gradio interface at runtime.

The OpenAI API key is not hard-coded in the program.

---

## Default Model

The default model name used in the code is:

```text
gpt-5.4-mini
```

Users may change the model name from the Gradio input panel depending on their API access.

---

## Example Queries

### Single-Day Query

```text
What is the evapotranspiration level in Gangtok, India today?
```

### Date-Range Query

```text
Give actual evapotranspiration and ET0 for Ludhiana, India from 2026-05-01 to 2026-05-05.
```

### Future Query

```text
Analyze ET0 and irrigation implications for Delhi, India tomorrow.
```

### Past Query

```text
Give evapotranspiration data for Jaipur, India yesterday.
```

### Multi-Line Query

```text
Please analyze evapotranspiration for Nagpur, India.

I need actual evapotranspiration, ET0, and agricultural interpretation
from 2026-05-01 to 2026-05-05.
```

---

## Suggested Indian Agricultural Test Locations

The following locations may be useful for representative agricultural evapotranspiration analysis across India:

```python
india_agri_locations = [
    "Ludhiana, India",
    "Karnal, India",
    "Lucknow, India",
    "Patna, India",
    "Kolkata, India",
    "Guwahati, India",
    "Gangtok, India",
    "Jaipur, India",
    "Jodhpur, India",
    "Ahmedabad, India",
    "Nagpur, India",
    "Bhopal, India",
    "Hyderabad, India",
    "Bengaluru, India",
    "Coimbatore, India",
]
```

These locations provide broad representative coverage of:

* Indo-Gangetic plains,
* semi-arid western India,
* arid Rajasthan,
* central India,
* Deccan plateau,
* humid eastern India,
* Himalayan foothills,
* coastal and southern agricultural regions,
* and Northeast India.

---

## Output Tabs in the Gradio Interface

The Gradio interface contains the following major tabs:

| Tab                            | Purpose                                                         |
| ------------------------------ | --------------------------------------------------------------- |
| GPT-Generated Report           | Displays the scientific GPT-assisted evapotranspiration report  |
| Chart                          | Displays the generated evapotranspiration and temperature chart |
| Daily Summary Data             | Shows daily aggregated metrics and provides CSV download        |
| Evaluation and Latency Metrics | Shows API delay, processing time, model, and query metadata     |
| Hourly Raw Data CSV            | Provides downloadable hourly raw Open-Meteo data                |

---

## CSV Output Details

### 1. Hourly Raw Data CSV

Contains hourly values such as:

* time,
* temperature,
* humidity,
* precipitation,
* wind speed,
* radiation,
* vapour pressure deficit,
* actual evapotranspiration,
* FAO reference ET₀,
* run ID,
* timestamp,
* place,
* query start date,
* query end date.

### 2. Daily Summary CSV

Contains daily aggregated values such as:

* daily actual evapotranspiration sum,
* daily ET₀ sum,
* daily precipitation sum,
* daily mean temperature,
* daily mean humidity,
* daily wind speed,
* daily vapour pressure deficit,
* daily shortwave radiation.

### 3. Evaluation and Latency Metrics CSV

Contains system-level metadata such as:

* original query,
* parsed place,
* start date,
* end date,
* date intent,
* query type,
* latitude,
* longitude,
* country,
* Open-Meteo endpoint used,
* OpenAI interpretation delay,
* geocoding delay,
* weather API delay,
* chart generation delay,
* GPT report generation delay,
* total delay,
* model used.

---

## Cumulative CSV Storage Logic

The system uses a cumulative CSV-saving strategy.

Instead of saving only the latest run, EvapoGPT:

1. reads the existing cumulative CSV if available,
2. prepares new output rows with run metadata,
3. concatenates old and new rows,
4. rewrites the full cumulative CSV,
5. creates a fresh downloadable snapshot for Gradio.

This avoids the common issue where only the most recent query output appears in the downloaded CSV.

---

## Important Limitations

EvapoGPT is a research prototype and should not be treated as a field-calibrated irrigation decision system.

Important cautions:

* Open-Meteo data are model-derived weather estimates.
* Evapotranspiration values may vary depending on model resolution and local terrain.
* Some variables may be unavailable or missing for certain locations or dates.
* Forecast data are limited to the supported forecast horizon.
* Date ranges are limited for stable dashboard operation.
* Agricultural decisions should be validated using local field observations, soil data, crop type, crop coefficient, and irrigation infrastructure.

---

## Research Use Cases

EvapoGPT may be useful for:

* agricultural water-demand analysis,
* evapotranspiration monitoring,
* irrigation advisory prototyping,
* agro-climatic comparison across regions,
* hydrological education,
* AI-assisted environmental dashboards,
* teaching Open-Meteo API integration,
* GPT-assisted scientific report generation,
* environmental data visualization,
* cumulative CSV-based research logging.

---

## Example Workflow

The internal workflow of EvapoGPT is:

```text
User Prompt
    ↓
OpenAI Query Interpretation
    ↓
Evapotranspiration Scope Check
    ↓
Location Extraction
    ↓
Open-Meteo Geocoding
    ↓
Date-Aware API Routing
    ↓
Open-Meteo Forecast / Archive Data Retrieval
    ↓
Hourly Data Cleaning
    ↓
Daily Aggregation
    ↓
Summary Metric Computation
    ↓
Chart Generation
    ↓
GPT-Based Scientific Report
    ↓
Cumulative CSV Storage
    ↓
Gradio Dashboard Output
```

---

## Technical Architecture

```text
+---------------------+
| User Natural Prompt |
+----------+----------+
           |
           v
+-----------------------------+
| OpenAI Prompt Interpretation |
+----------+------------------+
           |
           v
+-----------------------------+
| ET Scope and Date Validation |
+----------+------------------+
           |
           v
+-----------------------------+
| Open-Meteo Geocoding API     |
+----------+------------------+
           |
           v
+-----------------------------+
| Forecast / Archive API       |
+----------+------------------+
           |
           v
+-----------------------------+
| Pandas Data Processing       |
+----------+------------------+
           |
           v
+-----------------------------+
| Summary + Visualization      |
+----------+------------------+
           |
           v
+-----------------------------+
| OpenAI Scientific Report     |
+----------+------------------+
           |
           v
+-----------------------------+
| Gradio Dashboard + CSV Files |
+-----------------------------+
```

---

## Recommended Citation

If you use this tool in academic or research work, cite it as:

```text
Ray, P. P. (2026). EvapoGPT: GPT API-Assisted Evapotranspiration Analysis Dashboard. 
Department of Computer Applications, Sikkim University, India.
```

---

## Author

**Partha Pratim Ray**
Assistant Professor
Department of Computer Applications
Sikkim University, India

---

## License

This project may be released under an open-source license such as the MIT License.

Suggested license:

```text
MIT License
```

You may include a separate `LICENSE` file in the repository.

---

## Disclaimer

EvapoGPT is intended for research, education, and prototype-level environmental data analysis. The generated reports and values should not be used as the sole basis for irrigation scheduling, agricultural investment, drought declaration, or official hydrological decision-making.

For operational agricultural use, combine EvapoGPT outputs with:

* local meteorological station data,
* soil moisture readings,
* crop type,
* crop coefficient values,
* irrigation method,
* field observations,
* and expert agronomic judgment.

```
```
