# TravelHunters 🌍✈️

An intelligent travel recommendation system that suggests hotels based on user preferences. The project uses web scraping from Booking.com to collect comprehensive hotel data and generate personalized recommendations using machine learning.

## Project Overview

TravelHunters combines multiple data sources:

- **Hotels**: Over 8,000 hotels from Booking.com with images and ratings
- **Destinations**: Comprehensive destination information with Wikipedia integration

The system provides personalized recommendations through:

- Content-Based Filtering based on hotel attributes
- Collaborative Filtering for user behavior patterns
- Hybrid approaches for optimal recommendation quality

## Features

### Data Acquisition

- **Web Scraping**: Automated scraping of hotel data from Booking.com
- **Image Download**: Bulk download and processing of hotel images
- **Wikipedia Integration**: Destination information and images from Wikipedia
- **Data Validation**: Quality checks and data cleaning pipelines

### Machine Learning Models
- **Content-Based Filtering**: Recommendations based on item features
- **Collaborative Filtering**: User behavior-based recommendations
- **Hybrid Models**: Combined approaches for enhanced accuracy
- **Evaluation Metrics**: Comprehensive model performance assessment

### Database & Storage

- **SQLite Database**: Structured storage for hotels and destinations
- **Image Management**: Organized storage with automated naming conventions
- **Data Versioning**: Backup and versioning of scraped data

## Project Structure

TravelHunters follows the Data Science project structure with these main directories:

- `data_acquisition/` - Web scraping scripts and data collection tools
- `database/` - SQLite database and schema definitions
- `eda/` - Exploratory Data Analysis notebooks and reports
- `modelling/` - Machine Learning models and evaluation scripts
- `evaluation/` - Model performance analysis and metrics
- `docs/` - Project documentation and reports

### Key Files

- `scraping_data_files/` - Scrapy spiders for data collection
- `data_acquisition/download_json_images.py` - Image download automation
- `database/travelhunters.db` - Main SQLite database
- `conda.yml` - Python environment configuration

## Getting Started

### Prerequisites

- Python 3.8+
- Conda package manager
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TravelHunters
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f conda.yml
   conda activate travelhunters
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and configurations
   ```

### Usage

#### Data Collection

1. **Scrape hotel data from Booking.com**:
   ```bash
   cd scraping_data_files
   scrapy crawl booking_spider
   ```

2. **Download images**:
   ```bash
   cd data_acquisition
   python download_json_images.py
   ```

3. **Merge and process data**:
   ```bash
   cd mergingjson
   python merging_json_booking.py
   python merging_json_activity.py
   ```

#### Model Training and Evaluation

1. **Run EDA**:
   ```bash
   cd eda
   python generate-data-profile.py
   ```

2. **Train models**:
   ```bash
   cd modelling
   # Run your model training scripts
   ```

3. **Evaluate performance**:
   ```bash
   cd evaluation
   # Run evaluation scripts
   ```

## Data Sources

### Hotels (Booking.com)
- **Volume**: 8,000+ hotels
- **Attributes**: Name, location, rating, price, amenities, images
- **Coverage**: Major tourist destinations worldwide


### Destinations
- **Source**: Wikipedia integration
- **Content**: Destination descriptions, images, geographical data
- **Coverage**: Comprehensive location information

## Documentation

The project includes comprehensive documentation built with Quarto:

- **Project Charter**: Initial project scope and objectives
- **Data Report**: Data collection and quality analysis  
- **Modeling Report**: ML model development and selection
- **Evaluation Report**: Performance metrics and results

Build documentation:
```bash
cd docs
quarto render
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Team

**Data Science Summer School 2025 - ZHAW School of Engineering**

- Leona Kryeziu
- Evan Blazo
- Joan Felber
- Jakub Baranec

## References

- [Python Development Guide](refs/python_dev_guide.md)
- Booking.com API Documentation
- Scrapy Documentation
- Scikit-learn Documentation