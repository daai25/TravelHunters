import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [language, setLanguage] = useState("de");
  const [darkMode, setDarkMode] = useState(false);
  const [inputText, setInputText] = useState("");
  const [uploadedImages, setUploadedImages] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [cityPrediction, setCityPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  
  // Updated API endpoint for unified travel API
  const [apiEndpoint] = useState("http://localhost:5002/travel_recommendations");

  // Debug: Log API calls
  useEffect(() => {
    console.log("🔗 Unified Travel API Endpoint:", apiEndpoint);
  }, [apiEndpoint]);

  // Handle scroll effect for navigation
  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 50;
      setScrolled(isScrolled);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const toggleLanguage = () => {
    setLanguage((prev) => (prev === "de" ? "en" : "de"));
  };

  const toggleDarkMode = () => {
    setDarkMode((prev) => !prev);
  };

  // Handle image upload
  const handleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    const validImages = files.filter(file => file.type.startsWith('image/'));
    
    if (validImages.length !== files.length) {
      alert(language === "de" ? 
        "Bitte wählen Sie nur Bilddateien aus!" : 
        "Please select only image files!"
      );
    }

    validImages.forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const newImage = {
          id: Date.now() + Math.random(),
          file: file,
          preview: e.target.result,
          name: file.name
        };
        setUploadedImages(prev => [...prev, newImage]);
      };
      reader.readAsDataURL(file);
    });
  };

  // Remove uploaded image
  const removeImage = (imageId) => {
    setUploadedImages(prev => prev.filter(img => img.id !== imageId));
  };

  // Handle booking button click - opens hotel link from database
  const handleBooking = (hotel) => {
    console.log("🎯 Booking button clicked for hotel:", hotel);
    console.log("🔗 Available link fields:", {
      link: hotel.link,
      url: hotel.url, 
      booking_url: hotel.booking_url
    });

    // Check if hotel has a link/URL from database - try multiple fields
    const possibleLinks = [
      hotel.link,
      hotel.url, 
      hotel.booking_url,
      hotel.website,
      hotel.hotel_url
    ];

    let hotelUrl = null;
    for (const linkField of possibleLinks) {
      if (linkField && typeof linkField === 'string' && linkField.trim()) {
        hotelUrl = linkField.trim();
        console.log("✅ Found valid link:", hotelUrl);
        break;
      }
    }

    if (hotelUrl) {
      // Ensure URL has protocol
      if (!hotelUrl.startsWith('http://') && !hotelUrl.startsWith('https://')) {
        hotelUrl = 'https://' + hotelUrl;
      }
      
      console.log("🚀 Opening URL:", hotelUrl);
      // Open hotel link in new tab
      window.open(hotelUrl, '_blank', 'noopener,noreferrer');
    } else {
      // Fallback if no link available - try generic booking site search
      const searchQuery = encodeURIComponent(`${hotel.name} ${hotel.location}`);
      const fallbackUrl = `https://www.booking.com/search.html?ss=${searchQuery}`;
      
      console.log("❌ No direct link found, using booking.com search:", fallbackUrl);
      window.open(fallbackUrl, '_blank', 'noopener,noreferrer');
    }
  };

  const fetchRecommendations = async () => {
    // Check what inputs are provided
    const hasText = inputText.trim().length > 0;
    const hasImages = uploadedImages.length > 0;
    
    if (!hasText && !hasImages) {
      alert(language === "de" ? 
        "Bitte geben Sie entweder Text ein oder laden Sie ein Bild hoch!" : 
        "Please enter text or upload an image!"
      );
      return;
    }

    setIsLoading(true);
    setCityPrediction(null);
    setRecommendations([]);
    
    console.log("🚀 Starting recommendation with:", { hasText, hasImages });
    
    try {
      let apiUrl, formData;
      
      if (hasText && hasImages) {
        // ✅ BOTH: Use complete pipeline (image + text)
        console.log("🔄 Using complete pipeline (image + text)");
        apiUrl = "http://localhost:5002/travel_recommendations";
        
        formData = new FormData();
        formData.append('query', inputText.trim());
        formData.append('image', uploadedImages[0].file);
        
      } else if (hasImages && !hasText) {
        // 🏙️ IMAGE ONLY: Just predict city, then use generic hotel search
        console.log("🔄 Using image-only prediction");
        apiUrl = "http://localhost:5002/predict_city";
        
        formData = new FormData();
        formData.append('image', uploadedImages[0].file);
        
      } else if (hasText && !hasImages) {
        // 📝 TEXT ONLY: Just hotel recommendations
        console.log("🔄 Using text-only hotel search");
        apiUrl = "http://localhost:5002/recommend_hotels";
        
        // For hotel-only API, we need JSON
        formData = null; // Will use JSON instead
      }

      console.log("📡 Sending request to:", apiUrl);
      
      let response;
      if (formData) {
        // Multipart form data (for image uploads)
        response = await fetch(apiUrl, {
          method: 'POST',
          body: formData,
        });
      } else {
        // JSON data (for text-only)
        response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query: inputText.trim() })
        });
      }
      
      console.log("📥 Response status:", response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error("❌ HTTP Error Response:", errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("📊 API Response:", data);
      
      if (data.success) {
        // Handle different response types
        if (hasText && hasImages) {
          // Complete pipeline response
          handleCompleteResponse(data);
        } else if (hasImages && !hasText) {
          // City prediction only - then do hotel search
          await handleImageOnlyResponse(data);
        } else if (hasText && !hasImages) {
          // Hotel recommendations only
          handleTextOnlyResponse(data);
        }
      } else {
        throw new Error(data.error || "API returned unsuccessful response");
      }
      
    } catch (error) {
      console.error('❌ API Error:', error);
      
      let errorMessage;
      if (error.message.includes('Failed to fetch')) {
        errorMessage = language === "de" ? 
          "Verbindungsfehler: Ist der API-Server auf Port 5002 gestartet?" : 
          "Connection error: Is the API server running on port 5002?";
      } else {
        errorMessage = language === "de" ? 
          `Fehler: ${error.message}` : 
          `Error: ${error.message}`;
      }
      
      alert(errorMessage);
      setCityPrediction(null);
      setRecommendations([]);
    }
    
    setIsLoading(false);
  };

  // Handle complete pipeline response (text + image)
  const handleCompleteResponse = (data) => {
    if (data.city_prediction) {
      setCityPrediction({
        city: data.city_prediction.city,
        confidence: data.city_prediction.confidence,
        thresholdMet: data.query.confidence_threshold_met,
        originalQuery: data.query.original,
        modifiedQuery: data.query.modified
      });
    }

    if (data.hotel_recommendations && data.hotel_recommendations.length > 0) {
      setRecommendations(formatHotelRecommendations(data.hotel_recommendations));
    }
  };

  // Handle image-only response - predict city then search hotels
  const handleImageOnlyResponse = async (data) => {
    if (data.prediction) {
      const cityData = {
        city: data.prediction.city,
        confidence: data.prediction.confidence,
        thresholdMet: false,
        originalQuery: "",
        modifiedQuery: ""
      };
      setCityPrediction(cityData);

      // Now search for hotels in the predicted city
      try {
        console.log("🏨 Searching hotels for predicted city:", data.prediction.city);
        const hotelQuery = language === "de" ? 
          `Hotels in ${data.prediction.city}` : 
          `Hotels in ${data.prediction.city}`;
        
        const hotelResponse = await fetch("http://localhost:5002/recommend_hotels", {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query: hotelQuery })
        });

        if (hotelResponse.ok) {
          const hotelData = await hotelResponse.json();
          if (hotelData.success && hotelData.recommendations) {
            setRecommendations(formatHotelRecommendations(hotelData.recommendations));
          }
        }
      } catch (error) {
        console.error("❌ Error fetching hotels for predicted city:", error);
      }
    }
  };

  // Handle text-only response
  const handleTextOnlyResponse = (data) => {
    if (data.recommendations && data.recommendations.length > 0) {
      setRecommendations(formatHotelRecommendations(data.recommendations));
    }
  };

  // Helper function to format hotel recommendations
  const formatHotelRecommendations = (hotels) => {
    return hotels.map(hotel => ({
      id: hotel.id || hotel.rank,
      name: hotel.name,
      location: hotel.location,
      rating: hotel.rating,
      price: hotel.price,
      image: hotel.image || "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=400&h=300&fit=crop",
      description: hotel.description,
      amenities: hotel.amenities || ["WiFi", "Service"],
      link: hotel.link || hotel.url || hotel.booking_url || null,
      similarity_score: hotel.similarity_score,
      rank: hotel.rank
    }));
  };

  const clearSearch = () => {
    setInputText("");
    setUploadedImages([]);
    setRecommendations([]);
    setCityPrediction(null);
  };

  const scrollToSearch = () => {
    document.getElementById('search-section').scrollIntoView({ 
      behavior: 'smooth' 
    });
  };

  const scrollToAbout = () => {
    document.getElementById('about-section').scrollIntoView({ 
      behavior: 'smooth' 
    });
  };

  const renderStars = (rating) => {
    // Just return single star with rating number
    return (
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <span style={{ fontSize: "1.25rem" }}>⭐</span>
        <span style={{ fontWeight: "600", color: "var(--text-dark)" }}>{rating}</span>
      </div>
    );
  };

  const translations = {
    de: {
      title: "Travel Hunters",
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
      loading: "Wir analysieren Ihre Anfrage und suchen die besten Hotels für Sie...",
      perNight: "pro Nacht",
      bookNow: "Jetzt buchen",
      footerTitle: "Travel Hunters",
      footerDescription: "Data Science Summer School 2025 – ZHAW School of Engineering",
      team: "Ein Projekt von: Leona Kryeziu, Evan Blazo, Jolan Felber, Jakub Baranec",
      uploadImages: "Bilder hochladen (erforderlich)",
      uploadDescription: "Laden Sie ein Bild Ihres Traumreiseziels hoch - unsere KI erkennt die Stadt",
      cityPrediction: "Erkannte Stadt",
      confidence: "Sicherheit",
      queryModified: "Suchanfrage wurde erweitert",
      bothRequired: "Sowohl Text als auch Bild sind erforderlich"
    },
    en: {
      title: "Travel Hunters",
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
      loading: "We're analyzing your request and finding the best hotels for you...",
      perNight: "per night",
      bookNow: "Book Now",
      footerTitle: "Travel Hunters",
      footerDescription: "Data Science Summer School 2025 – ZHAW School of Engineering",
      team: "A project by: Leona Kryeziu, Evan Blazo, Jolan Felber, Jakub Baranec",
      uploadImages: "Upload Images (required)",
      uploadDescription: "Upload an image of your dream destination - our AI will recognize the city",
      cityPrediction: "Detected City",
      confidence: "Confidence",
      queryModified: "Search query was enhanced",
      bothRequired: "Both text and image are required"
    }
  };

  const t = translations[language];

  return (
    <div className={`app ${darkMode ? "dark" : ""}`}>
      {/* Navigation */}
      <nav className={`nav ${scrolled ? "scrolled" : ""}`}>
        <div className="logo" style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary-color)' }}>
          🧭 {t.title}
        </div>
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
            <button className="btn btn-secondary" onClick={scrollToAbout}>
              📖 {t.learnMore}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main>
        {/* About Section */}
        <section id="about-section" className="section">
          <div className="search-section">
            <h2 className="section-title">
              {language === "de" ? "Warum Travel Hunters?" : "Why Travel Hunters?"}
            </h2>
            <div style={{ maxWidth: "800px", margin: "0 auto", textAlign: "center" }}>
              <p style={{ fontSize: "1.25rem", marginBottom: "2.5rem", color: "var(--text-dark)", fontWeight: "500" }}>
                {language === "de" 
                  ? "Verwandeln Sie Ihre Reiseträume in unvergessliche Erlebnisse - mit der Kraft des Machine Learning."
                  : "Transform your travel dreams into unforgettable experiences - powered by machine learning."
                }
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "3rem", marginTop: "3rem" }}>
                <div style={{ textAlign: "center", padding: "2rem 1rem" }}>
                  <div style={{ fontSize: "4rem", marginBottom: "1.5rem", filter: "drop-shadow(0 4px 8px rgba(0,0,0,0.1))" }}>✨</div>
                  <h3 style={{ marginBottom: "1rem", color: "var(--primary-color)", fontSize: "1.5rem", fontWeight: "700" }}>
                    {language === "de" ? "Magische Entdeckungen" : "Magical Discoveries"}
                  </h3>
                  <p style={{ color: "var(--text-light)", fontSize: "1.1rem", lineHeight: "1.7" }}>
                    {language === "de" 
                      ? "Zeigen Sie uns ein Bild Ihres Traumziels - unser CNN-Modell erkennt sofort, wohin Ihr Herz Sie führt und findet die perfekten Hotels dafür."
                      : "Show us a picture of your dream destination - our CNN model instantly recognizes where your heart wants to go and finds the perfect hotels for it."
                    }
                  </p>
                </div>
                <div style={{ textAlign: "center", padding: "2rem 1rem" }}>
                  <div style={{ fontSize: "4rem", marginBottom: "1.5rem", filter: "drop-shadow(0 4px 8px rgba(0,0,0,0.1))" }}>🎯</div>
                  <h3 style={{ marginBottom: "1rem", color: "var(--primary-color)", fontSize: "1.5rem", fontWeight: "700" }}>
                    {language === "de" ? "Perfekte Matches" : "Perfect Matches"}
                  </h3>
                  <p style={{ color: "var(--text-light)", fontSize: "1.1rem", lineHeight: "1.7" }}>
                    {language === "de" 
                      ? "Beschreiben Sie Ihre Reiseträume in Ihren eigenen Worten - wir verstehen sie und finden Hotels, die genau zu Ihnen passen."
                      : "Describe your travel dreams in your own words - we understand them and find hotels that match you perfectly."
                    }
                  </p>
                </div>
                <div style={{ textAlign: "center", padding: "2rem 1rem" }}>
                  <div style={{ fontSize: "4rem", marginBottom: "1.5rem", filter: "drop-shadow(0 4px 8px rgba(0,0,0,0.1))" }}>🌍</div>
                  <h3 style={{ marginBottom: "1rem", color: "var(--primary-color)", fontSize: "1.5rem", fontWeight: "700" }}>
                    {language === "de" ? "Weltweite Auswahl" : "Global Selection"}
                  </h3>
                  <p style={{ color: "var(--text-light)", fontSize: "1.1rem", lineHeight: "1.7" }}>
                    {language === "de" 
                      ? "Von den Malediven bis nach New York - entdecken Sie über 127 Destinationen weltweit mit tausenden handverlesenen Hotels."
                      : "From the Maldives to New York - discover over 127 destinations worldwide with thousands of hand-picked hotels."
                    }
                  </p>
                </div>
                <div style={{ textAlign: "center", padding: "2rem 1rem" }}>
                  <div style={{ fontSize: "4rem", marginBottom: "1.5rem", filter: "drop-shadow(0 4px 8px rgba(0,0,0,0.1))" }}>⚡</div>
                  <h3 style={{ marginBottom: "1rem", color: "var(--primary-color)", fontSize: "1.5rem", fontWeight: "700" }}>
                    {language === "de" ? "Sofortige Inspiration" : "Instant Inspiration"}
                  </h3>
                  <p style={{ color: "var(--text-light)", fontSize: "1.1rem", lineHeight: "1.7" }}>
                    {language === "de" 
                      ? "Ein Klick, unendliche Möglichkeiten. Von der Idee bis zur Buchung - entdecken Sie Ihr nächstes Abenteuer in Sekunden."
                      : "One click, endless possibilities. From idea to booking - discover your next adventure in seconds."
                    }
                  </p>
                </div>
              </div>
              
              {/* Call to Action */}
              <div style={{ 
                marginTop: "3rem", 
                padding: "2.5rem", 
                background: "linear-gradient(135deg, rgba(37, 99, 235, 0.05) 0%, rgba(79, 70, 229, 0.05) 100%)",
                borderRadius: "20px",
                border: "1px solid rgba(37, 99, 235, 0.1)"
              }}>
                <h3 style={{ color: "var(--primary-color)", marginBottom: "1rem", fontSize: "1.4rem" }}>
                  {language === "de" ? "Bereit für Ihr nächstes Abenteuer?" : "Ready for your next adventure?"}
                </h3>
                <p style={{ color: "var(--text-light)", marginBottom: "2rem", fontSize: "1.1rem" }}>
                  {language === "de" 
                    ? "Lassen Sie sich von der Magie des Machine Learning zu Ihrem perfekten Reiseziel führen."
                    : "Let the magic of machine learning guide you to your perfect destination."
                  }
                </p>
                <button 
                  className="btn"
                  onClick={scrollToSearch}
                  style={{ 
                    padding: "1rem 2rem",
                    fontSize: "1.1rem",
                    background: "linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%)",
                    boxShadow: "0 8px 25px rgba(37, 99, 235, 0.3)"
                  }}
                >
                  🚀 {language === "de" ? "Jetzt entdecken" : "Discover Now"}
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* Search Section */}
        <section id="search-section" className="section">
          <div className="search-section">
            <h2 className="section-title">{t.searchTitle}</h2>
            
            {/* Important notice - Updated */}
            <div style={{ 
              background: "var(--bg-secondary)", 
              border: "2px solid var(--primary-color)", 
              borderRadius: "12px", 
              padding: "1rem", 
              marginBottom: "2rem", 
              textAlign: "center" 
            }}>
              <p style={{ color: "var(--primary-color)", fontWeight: "600", margin: 0 }}>
                {language === "de" ? 
                  "💡 Geben Sie Text ein, laden Sie ein Bild hoch, oder beides für beste Ergebnisse!" :
                  "💡 Enter text, upload an image, or both for best results!"
                }
              </p>
            </div>

            <div className="search-form">
              <div className="input-group">
                <label htmlFor="interests" className="input-label">
                  {language === "de" ? "Ihre Reiseinteressen (optional wenn Bild hochgeladen)" : "Your Travel Interests (optional if image uploaded)"}
                </label>
                <textarea
                  id="interests"
                  placeholder={t.searchPlaceholder}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={isLoading}
                />
              </div>

              {/* Image Upload Section */}
              <div className="input-group">
                <label className="input-label">
                  {language === "de" ? "Bilder hochladen (optional wenn Text eingegeben)" : "Upload Images (optional if text provided)"}
                </label>
                <p className="upload-description">
                  {t.uploadDescription}
                </p>
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleImageUpload}
                  disabled={isLoading}
                  className="file-input"
                  key={uploadedImages.length === 0 ? 'empty' : 'has-files'} // Force re-render
                />
                
                {/* Display uploaded images */}
                {uploadedImages.length > 0 && (
                  <div className="image-grid">
                    {uploadedImages.map((image, index) => (
                      <div key={image.id} className="image-preview">
                        <img 
                          src={image.preview} 
                          alt={image.name}
                          className="preview-image"
                        />
                        <button 
                          onClick={() => removeImage(image.id)}
                          className="remove-button"
                          disabled={isLoading}
                        >
                          ✕
                        </button>
                        <p className="image-name">
                          {image.name} {index === 0 && <span style={{color: 'var(--primary-color)'}}>(Primary)</span>}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div style={{ display: "flex", gap: "1rem", justifyContent: "center" }}>
                <button 
                  onClick={fetchRecommendations}
                  disabled={isLoading || (!inputText.trim() && uploadedImages.length === 0)}
                  className="btn"
                >
                  {isLoading ? "🔄" : "🔍"} {t.showRecommendations}
                </button>
                {(recommendations.length > 0 || inputText || uploadedImages.length > 0 || cityPrediction) && (
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

        {/* City Prediction Results */}
        {cityPrediction && !isLoading && (
          <section className="section">
            <div className="search-section">
              <h2 className="section-title">🏙️ {t.cityPrediction}</h2>
              <div style={{ 
                background: "var(--bg-secondary)", 
                borderRadius: "12px", 
                padding: "1.5rem", 
                textAlign: "center",
                border: "1px solid var(--border-color)"
              }}>
                <h3 style={{ color: "var(--primary-color)", marginBottom: "1rem" }}>
                  {cityPrediction.city}
                </h3>
                <p style={{ color: "var(--text-light)", marginBottom: "1rem" }}>
                  {t.confidence}: {(cityPrediction.confidence * 100).toFixed(1)}%
                </p>
                {cityPrediction.thresholdMet && (
                  <div style={{ 
                    background: "var(--primary-color)", 
                    color: "white", 
                    padding: "0.5rem 1rem", 
                    borderRadius: "8px",
                    display: "inline-block"
                  }}>
                    ✅ {t.queryModified}
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* Results Section */}
        {recommendations.length > 0 && !isLoading && (
          <section className="section">
            <h2 className="section-title">🏨 {t.recommendations}</h2>
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
                      ⭐ {hotel.rating}
                    </div>
                    {hotel.rank && (
                      <div className="card-badge" style={{ 
                        top: '10px', 
                        left: '10px', 
                        background: 'var(--primary-color)' 
                      }}>
                        #{hotel.rank}
                      </div>
                    )}
                  </div>
                  <div className="card-content">
                    <h3 className="card-title">{hotel.name}</h3>
                    <p className="card-location">
                      📍 {hotel.location}
                    </p>
                    {/* Price moved above rating */}
                    <div style={{ 
                      fontSize: "1.25rem", 
                      fontWeight: "600", 
                      color: "var(--primary-color)", 
                      marginBottom: "0.75rem" 
                    }}>
                      💰 {hotel.price} {t.perNight}
                    </div>
                    <div className="rating">
                      {renderStars(hotel.rating)}
                      {hotel.similarity_score && (
                        <span style={{ marginLeft: "1rem", color: "var(--primary-color)", fontSize: "0.8rem" }}>
                          Match: {(hotel.similarity_score * 100).toFixed(0)}%
                        </span>
                      )}
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
                    <button 
                      className="btn" 
                      style={{ width: "100%" }}
                      onClick={() => handleBooking(hotel)}
                    >
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