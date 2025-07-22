// Professional TravelHunters App.js
import React, { useState, useEffect } from "react";
import "./App.css";
import hotel1 from './assets/hotel1.jpeg'; 
import hotel2 from "./assets/hotel2.jpg";

function App() {
  const [language, setLanguage] = useState("de");
  const [darkMode, setDarkMode] = useState(false);
  const [inputText, setInputText] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  // Handle scroll effect for navigation
  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 50;
      setScrolled(isScrolled);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Load saved preferences
  useEffect(() => {
    const savedLanguage = localStorage.getItem("travelHunters_language");
    const savedDarkMode = localStorage.getItem("travelHunters_darkMode");
    
    if (savedLanguage) setLanguage(savedLanguage);
    if (savedDarkMode) setDarkMode(JSON.parse(savedDarkMode));
  }, []);

  // Save preferences
  useEffect(() => {
    localStorage.setItem("travelHunters_language", language);
    localStorage.setItem("travelHunters_darkMode", JSON.stringify(darkMode));
  }, [language, darkMode]);

  const toggleLanguage = () => {
    setLanguage((prev) => (prev === "de" ? "en" : "de"));
  };

  const toggleDarkMode = () => {
    setDarkMode((prev) => !prev);
  };

  const fetchRecommendations = async () => {
    if (!inputText.trim()) {
      alert(language === "de" ? 
        "Bitte geben Sie Ihre Interessen ein!" : 
        "Please enter your interests!"
      );
      return;
    }

    setIsLoading(true);
    
    // Simulate API call with realistic delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const recommendationsData = [
      {
        id: 1,
        name: language === "de" ? "Hotel Sonnenblick" : "Hotel Sonnenblick",
        location: "Zürich, Schweiz",
        rating: 4.5,
        price: "CHF 280",
        image: hotel1,
        description: language === "de" 
          ? "Luxuriöses Hotel im Herzen von Zürich mit atemberaubender Aussicht"
          : "Luxurious hotel in the heart of Zurich with breathtaking views",
        amenities: language === "de" 
          ? ["Spa", "Restaurant", "Fitnessraum"] 
          : ["Spa", "Restaurant", "Fitness Center"]
      },
      {
        id: 2,
        name: language === "de" ? "Seehotel Panorama" : "Lake Hotel Panorama",
        location: "Luzern, Schweiz",
        rating: 4.8,
        price: "CHF 450",
        image: hotel2,
        description: language === "de"
          ? "Romantisches Hotel direkt am Vierwaldstättersee"
          : "Romantic hotel directly on Lake Lucerne",
        amenities: language === "de"
          ? ["Seeblick", "Wellness", "Gourmet Restaurant"]
          : ["Lake View", "Wellness", "Gourmet Restaurant"]
      },
      {
        id: 3,
        name: "Urban Stay Basel",
        location: "Basel, Schweiz",
        rating: 4.2,
        price: "CHF 180",
        image: null,
        description: language === "de"
          ? "Modernes Boutique-Hotel in der Kulturstadt Basel"
          : "Modern boutique hotel in the cultural city of Basel",
        amenities: language === "de"
          ? ["Zentrale Lage", "Coworking Space", "Rooftop Bar"]
          : ["Central Location", "Coworking Space", "Rooftop Bar"]
      },
      {
        id: 4,
        name: language === "de" ? "Bergresort Alpina" : "Alpine Resort Alpina",
        location: "Grindelwald, Schweiz",
        rating: 4.7,
        price: "CHF 380",
        image: null,
        description: language === "de"
          ? "Exklusives Resort mit direktem Zugang zu den Skipisten"
          : "Exclusive resort with direct access to ski slopes",
        amenities: language === "de"
          ? ["Ski-in/Ski-out", "Alpine Spa", "Bergpanorama"]
          : ["Ski-in/Ski-out", "Alpine Spa", "Mountain Panorama"]
      }
    ];
    
    setRecommendations(recommendationsData);
    setIsLoading(false);
  };

  const clearSearch = () => {
    setInputText("");
    setRecommendations([]);
  };

  const scrollToSearch = () => {
    document.getElementById('search-section').scrollIntoView({ 
      behavior: 'smooth' 
    });
  };

  const renderStars = (rating) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;

    for (let i = 0; i < fullStars; i++) {
      stars.push(<span key={`full-${i}`}>⭐</span>);
    }
    
    if (hasHalfStar) {
      stars.push(<span key="half">⭐</span>);
    }

    return stars;
  };

  const translations = {
    de: {
      title: "TravelHunters",
      subtitle: "Entdecken Sie Ihre nächste Traumreise",
      description: "Von den majestätischen Alpen bis zu pulsierenden Städten - finden Sie das perfekte Reiseziel für Ihr nächstes Abenteuer",
      searchButton: "Reise entdecken",
      learnMore: "Mehr erfahren",
      searchTitle: "Was interessiert Sie?",
      searchPlaceholder: "Beschreiben Sie Ihre Traumreise... (z.B. 'Romantische Städtereise', 'Abenteuer in den Bergen', 'Entspannung am See')",
      showRecommendations: "Empfehlungen anzeigen",
      clearSearch: "Suche löschen",
      recommendations: "Unsere Empfehlungen",
      noResults: "Keine Ergebnisse gefunden",
      loading: "Wir suchen die besten Optionen für Sie...",
      perNight: "pro Nacht",
      bookNow: "Jetzt buchen",
      footerTitle: "TravelHunters",
      footerDescription: "Data Science Summer School 2025 – ZHAW School of Engineering",
      team: "Ein Projekt von: Leona Kryeziu, Evan Blazo, Joan Felber, Jakub Baranec"
    },
    en: {
      title: "TravelHunters",
      subtitle: "Discover Your Next Dream Journey",
      description: "From majestic Alps to vibrant cities - find the perfect destination for your next adventure",
      searchButton: "Discover Travel",
      learnMore: "Learn More",
      searchTitle: "What interests you?",
      searchPlaceholder: "Describe your dream trip... (e.g., 'Romantic city break', 'Mountain adventure', 'Lakeside relaxation')",
      showRecommendations: "Show Recommendations",
      clearSearch: "Clear Search",
      recommendations: "Our Recommendations",
      noResults: "No results found",
      loading: "We're finding the best options for you...",
      perNight: "per night",
      bookNow: "Book Now",
      footerTitle: "TravelHunters",
      footerDescription: "Data Science Summer School 2025 – ZHAW School of Engineering",
      team: "A project by: Leona Kryeziu, Evan Blazo, Joan Felber, Jakub Baranec"
    }
  };

  const t = translations[language];

  return (
    <div className={`app ${darkMode ? "dark" : ""}`}>
      {/* Navigation */}
      <nav className={`nav ${scrolled ? "scrolled" : ""}`}>
        <a href="#" className="logo">
          🧭 {t.title}
        </a>
        <div className="nav-controls">
          <button 
            className="btn-icon" 
            onClick={toggleLanguage}
            title={language === "de" ? "Switch to English" : "Auf Deutsch wechseln"}
          >
            {language === "de" ? "🇬🇧" : "🇩🇪"}
          </button>
          <button 
            className="btn-icon" 
            onClick={toggleDarkMode}
            title={darkMode ? "Light Mode" : "Dark Mode"}
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="hero">
        <div className="hero-content">
          <h1>{t.title}</h1>
          <p>{t.subtitle}</p>
          <p style={{ fontSize: "1.125rem", opacity: 0.9, marginBottom: "2rem" }}>
            {t.description}
          </p>
          <div className="hero-cta">
            <button className="btn" onClick={scrollToSearch}>
              🔍 {t.searchButton}
            </button>
            <button className="btn btn-secondary">
              📖 {t.learnMore}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main>
        {/* Search Section */}
        <section id="search-section" className="section">
          <div className="search-section">
            <h2 className="section-title">{t.searchTitle}</h2>
            <div className="search-form">
              <div className="input-group">
                <label htmlFor="interests" className="input-label">
                  {language === "de" ? "Ihre Interessen" : "Your Interests"}
                </label>
                <textarea
                  id="interests"
                  placeholder={t.searchPlaceholder}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={isLoading}
                />
              </div>
              <div style={{ display: "flex", gap: "1rem", justifyContent: "center" }}>
                <button 
                  onClick={fetchRecommendations}
                  disabled={isLoading}
                  className="btn"
                >
                  {isLoading ? "🔄" : "🔍"} {t.showRecommendations}
                </button>
                {(recommendations.length > 0 || inputText) && (
                  <button 
                    onClick={clearSearch}
                    className="btn btn-secondary"
                    style={{ background: "var(--text-light)", color: "white" }}
                  >
                    🗑️ {t.clearSearch}
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Loading State */}
        {isLoading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>{t.loading}</p>
          </div>
        )}

        {/* Results Section */}
        {recommendations.length > 0 && !isLoading && (
          <section className="section">
            <h2 className="section-title">{t.recommendations}</h2>
            <div className="cards">
              {recommendations.map((hotel) => (
                <div className="card animate-fade-in-up" key={hotel.id}>
                  <div className="card-image">
                    {hotel.image ? (
                      <img 
                        src={hotel.image} 
                        alt={hotel.name} 
                        className="hotel-image" 
                      />
                    ) : (
                      <div className="placeholder-card">
                        🏨
                      </div>
                    )}
                    <div className="card-badge">
                      {hotel.price} {t.perNight}
                    </div>
                  </div>
                  <div className="card-content">
                    <h3 className="card-title">{hotel.name}</h3>
                    <p className="card-location">
                      📍 {hotel.location}
                    </p>
                    <div className="rating">
                      {renderStars(hotel.rating)} 
                      <span style={{ marginLeft: "0.5rem", color: "var(--text-light)" }}>
                        ({hotel.rating})
                      </span>
                    </div>
                    <p style={{ 
                      margin: "1rem 0", 
                      color: "var(--text-light)", 
                      fontSize: "0.9rem" 
                    }}>
                      {hotel.description}
                    </p>
                    <div style={{ 
                      display: "flex", 
                      flexWrap: "wrap", 
                      gap: "0.5rem", 
                      marginBottom: "1.5rem" 
                    }}>
                      {hotel.amenities.map((amenity, idx) => (
                        <span 
                          key={idx}
                          style={{
                            padding: "0.25rem 0.75rem",
                            background: "var(--bg-secondary)",
                            border: "1px solid var(--border-color)",
                            borderRadius: "20px",
                            fontSize: "0.8rem",
                            color: "var(--text-light)"
                          }}
                        >
                          {amenity}
                        </span>
                      ))}
                    </div>
                    <button className="btn" style={{ width: "100%" }}>
                      🎯 {t.bookNow}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer>
        <div className="footer-content">
          <h3 className="footer-title">🧭 {t.footerTitle}</h3>
          <p className="footer-text">{t.footerDescription}</p>
          <p className="footer-team">{t.team}</p>
        </div>
      </footer>
    </div>
  );
}

export default App;