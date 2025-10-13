# Stimulus Function Explorer

A comprehensive web application for exploring and discovering educational stimulus functions with integrated educational standards.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stimfunc.streamlit.app/)

## 📋 Features

- **Function Gallery**: Browse all available stimulus functions with visual examples
- **Advanced Search**: Search by function name, category, tags, description, or educational standards
- **Educational Standards Integration**: View CCSS and other educational standards linked to each function
- **AI-Powered Search**: Natural language search capabilities
- **Function Details**: Comprehensive information about each function including parameters, examples, and standards

## 🛠️ Local Development

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mauricio-zorzan/stimulus-function-explorer.git
cd stimulus-function-explorer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your database credentials
DB_USERNAME=your_username
DB_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key  # Optional, for AI search
```

4. Run the application:
```bash
streamlit run app_new.py
```

## 📊 Data Management

### Updating Educational Standards

To update the educational standards from the database:

```bash
python update_function_standards.py
```

This script will:
- Connect to the MySQL database
- Fetch standards for all functions
- Update the local JSON files with the latest standards data

## 🏗️ Project Structure

```
├── app_new.py                 # Main Streamlit application
├── src/                       # Source code modules
│   ├── database.py           # Database connection and queries
│   └── ai_search.py          # AI-powered search functionality
├── data/                      # Function data and images
│   ├── functions/            # JSON files for each function
│   └── images/               # Example images for functions
├── drawing_functions/         # Drawing function implementations
├── stimulus_descriptions/     # Stimulus type descriptions
└── requirements.txt          # Python dependencies
```

## 🎯 Educational Standards

The application integrates with educational standards databases to provide:
- **CCSS Standards**: Common Core State Standards for Mathematics
- **Function Mapping**: Links between functions and relevant standards
- **Search Integration**: Find functions by searching for specific standards

## 🔧 Configuration

### Database Connection

The app connects to a MySQL database to fetch educational standards. Configure your database credentials in a `.env` file:

```env
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

### AI Search (Optional)

For enhanced search capabilities, add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_key
```

## 📈 Usage Statistics

- **92 Functions**: Comprehensive collection of educational stimulus functions
- **78 Functions with Standards**: Functions linked to educational standards
- **Multiple Categories**: Geometry, data visualization, number operations, and more

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for educational technology**