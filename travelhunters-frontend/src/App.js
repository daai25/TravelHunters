import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [language, setLanguage] = useState("de");
  const [darkMode, setDarkMode] = useState(false);
  const [inputText, setInputText] = useState("");
  const [uploadedImages, setUploadedImages] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  // Set your API endpoint here (hidden from users)
  const [apiEndpoint] = useState("http://localhost:8000/recommend");

  // Debug: Log API calls
  useEffect(() => {
    console.log("🔗 API Endpoint:", apiEndpoint);
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
      // Fallback if no link available
      console.log("❌ No valid link found for hotel:", hotel.name);
      const bookingMessage = language === "de" 
        ? `Leider ist kein direkter Buchungslink für ${hotel.name} verfügbar. Bitte besuchen Sie deren Website direkt.`
        : `Unfortunately, no direct booking link is available for ${hotel.name}. Please visit their website directly.`;
      
      alert(bookingMessage);
    }
  };

  const fetchRecommendations = async () => {
    if (!inputText.trim() && uploadedImages.length === 0) {
      alert(language === "de" ? 
        "Bitte geben Sie Ihre Interessen ein oder laden Sie Bilder hoch!" : 
        "Please enter your interests or upload images!"
      );
      return;
    }

    setIsLoading(true);
    console.log("🚀 Starting recommendation fetch...");
    console.log("📝 Text input:", inputText);
    console.log("📸 Images:", uploadedImages.length);
    console.log("🔗 API URL:", apiEndpoint);
    
    try {
      // FIXED: Use API if endpoint is set AND (text OR images are provided)
      if (apiEndpoint && (inputText.trim() || uploadedImages.length > 0)) {
        console.log("✅ Calling ML API...");
        
        const formData = new FormData();
        
        // Add text input (even if empty, API can handle it)
        formData.append('text_input', inputText.trim());
        formData.append('language', language);
        
        // Add images if any
        uploadedImages.forEach((image, index) => {
          formData.append(`image_${index}`, image.file);
          console.log(`📎 Added image_${index}:`, image.name);
        });

        try {
          console.log("📡 Sending request to:", apiEndpoint);
          const response = await fetch(apiEndpoint, {
            method: 'POST',
            body: formData,
          });
          
          console.log("📥 Response status:", response.status);
          
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const data = await response.json();
          console.log("📊 API Response:", data);
          
          if (data.recommendations && data.recommendations.length > 0) {
            // Convert ML response format to frontend format
            const formattedRecommendations = data.recommendations.map(hotel => ({
              id: hotel.id || hotel.rank,
              name: hotel.name,
              location: hotel.location,
              rating: hotel.rating,
              price: hotel.price, // Already formatted as "CHF 280"
              image: hotel.image || "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=400&h=300&fit=crop", // Fallback image
              description: hotel.description,
              amenities: hotel.amenities || ["WiFi", "Service"],
              // Add booking link from database
              link: hotel.link || hotel.url || hotel.booking_url || null
            }));
            
            setRecommendations(formattedRecommendations);
            console.log("✅ Set ML recommendations:", formattedRecommendations.length);
          } else {
            console.log("⚠️ No recommendations in API response, using fallback");
            await generateSampleRecommendations();
          }
        } catch (error) {
          console.error('❌ API Error:', error);
          alert(language === "de" ? 
            "Fehler beim Verarbeiten der Anfrage. Verwende Beispieldaten." : 
            "Error processing request. Using sample data."
          );
          // Fallback to sample data
          await generateSampleRecommendations();
        }
      } else {
        console.log("ℹ️ No API endpoint or input, using sample data");
        // Fallback to sample data when no API endpoint
        await generateSampleRecommendations();
      }
    } catch (error) {
      console.error('💥 General Error:', error);
      await generateSampleRecommendations();
    }
    
    setIsLoading(false);
  };

  const generateSampleRecommendations = async () => {
    console.log("🔄 Generating sample recommendations...");
    // Simulate API call with realistic delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const recommendationsData = [
      {
        id: 1,
        name: language === "de" ? "Hotel Sonnenblick" : "Hotel Sonnenblick",
        location: "Zürich, Schweiz",
        rating: 4.5,
        price: "CHF 280",
        image: "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=400&h=300&fit=crop",
        description: language === "de" 
          ? "Luxuriöses Hotel im Herzen von Zürich mit atemberaubender Aussicht"
          : "Luxurious hotel in the heart of Zurich with breathtaking views",
        amenities: language === "de" 
          ? ["Spa", "Restaurant", "Fitnessraum"] 
          : ["Spa", "Restaurant", "Fitness Center"],
        link: "https://www.booking.com"
      },
      {
        id: 2,
        name: language === "de" ? "Seehotel Panorama" : "Lake Hotel Panorama",
        location: "Luzern, Schweiz",
        rating: 4.8,
        price: "CHF 450",
        image: "https://images.unsplash.com/photo-1571896349842-33c89424de2d?w=400&h=300&fit=crop",
        description: language === "de"
          ? "Romantisches Hotel direkt am Vierwaldstättersee"
          : "Romantic hotel directly on Lake Lucerne",
        amenities: language === "de"
          ? ["Seeblick", "Wellness", "Gourmet Restaurant"]
          : ["Lake View", "Wellness", "Gourmet Restaurant"],
        link: "https://www.booking.com"
      },
      {
        id: 3,
        name: "Urban Stay Basel",
        location: "Basel, Schweiz",
        rating: 4.2,
        price: "CHF 180",
        image: "https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=400&h=300&fit=crop",
        description: language === "de"
          ? "Modernes Boutique-Hotel in der Kulturstadt Basel"
          : "Modern boutique hotel in the cultural city of Basel",
        amenities: language === "de"
          ? ["Zentrale Lage", "Coworking Space", "Rooftop Bar"]
          : ["Central Location", "Coworking Space", "Rooftop Bar"],
        link: "https://www.booking.com"
      },
      {
        id: 4,
        name: language === "de" ? "Bergresort Alpina" : "Alpine Resort Alpina",
        location: "Grindelwald, Schweiz",
        rating: 4.7,
        price: "CHF 380",
        image: "https://images.unsplash.com/photo-1551218808-94e220e084d2?w=400&h=300&fit=crop",
        description: language === "de"
          ? "Exklusives Resort mit direktem Zugang zu den Skipisten"
          : "Exclusive resort with direct access to ski slopes",
        amenities: language === "de"
          ? ["Ski-in/Ski-out", "Alpine Spa", "Bergpanorama"]
          : ["Ski-in/Ski-out", "Alpine Spa", "Mountain Panorama"],
        link: "https://www.booking.com"
      }
    ];
    
    setRecommendations(recommendationsData);
    console.log("✅ Set sample recommendations");
  };

  const clearSearch = () => {
    setInputText("");
    setUploadedImages([]);
    setRecommendations([]);
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
      loading: "Wir analysieren Ihre Anfrage und suchen die besten Optionen für Sie...",
      perNight: "pro Nacht",
      bookNow: "Jetzt buchen",
      footerTitle: "TravelHunters",
      footerDescription: "Data Science Summer School 2025 – ZHAW School of Engineering",
      team: "Ein Projekt von: Leona Kryeziu, Evan Blazo, Joan Felber, Jakub Baranec",
      uploadImages: "Bilder hochladen",
      uploadDescription: "Laden Sie Bilder hoch, die Ihre Reisevorstellungen zeigen",
      apiEndpoint: "API-Endpunkt",
      apiPlaceholder: "Pfad zu Ihrem Kollegen-Skript (z.B. http://localhost:5000/analyze)"
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
      loading: "We're analyzing your request and finding the best options for you...",
      perNight: "per night",
      bookNow: "Book Now",
      footerTitle: "TravelHunters",
      footerDescription: "Data Science Summer School 2025 – ZHAW School of Engineering",
      team: "A project by: Leona Kryeziu, Evan Blazo, Joan Felber, Jakub Baranec",
      uploadImages: "Upload Images",
      uploadDescription: "Upload images that represent your travel ideas",
      apiEndpoint: "API Endpoint",
      apiPlaceholder: "Path to your colleague's script (e.g., http://localhost:5000/analyze)"
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
              {language === "de" ? "Über TravelHunters" : "About TravelHunters"}
            </h2>
            <div style={{ maxWidth: "800px", margin: "0 auto", textAlign: "center" }}>
              <p style={{ fontSize: "1.125rem", marginBottom: "1.5rem", color: "var(--text-light)" }}>
                {language === "de" 
                  ? "TravelHunters ist eine intelligente Reiseempfehlungsplattform, die Ihnen hilft, das perfekte Reiseziel basierend auf Ihren persönlichen Interessen und hochgeladenen Bildern zu finden."
                  : "TravelHunters is an intelligent travel recommendation platform that helps you find the perfect destination based on your personal interests and uploaded images."
                }
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "2rem", marginTop: "2rem" }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🤖</div>
                  <h3 style={{ marginBottom: "0.5rem", color: "var(--text-dark)" }}>
                    {language === "de" ? "KI-Powered" : "AI-Powered"}
                  </h3>
                  <p style={{ color: "var(--text-light)" }}>
                    {language === "de" 
                      ? "Intelligente Algorithmen analysieren Ihre Präferenzen und Bilder"
                      : "Smart algorithms analyze your preferences and images"
                    }
                  </p>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📸</div>
                  <h3 style={{ marginBottom: "0.5rem", color: "var(--text-dark)" }}>
                    {language === "de" ? "Bilderkennung" : "Image Recognition"}
                  </h3>
                  <p style={{ color: "var(--text-light)" }}>
                    {language === "de" 
                      ? "Erkennung von Reisevorstellungen aus Ihren Bildern"
                      : "Recognition of travel ideas from your images"
                    }
                  </p>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>⚡</div>
                  <h3 style={{ marginBottom: "0.5rem", color: "var(--text-dark)" }}>
                    {language === "de" ? "Schnell & Einfach" : "Fast & Simple"}
                  </h3>
                  <p style={{ color: "var(--text-light)" }}>
                    {language === "de" 
                      ? "Sofortige Ergebnisse mit wenigen Klicks"
                      : "Instant results with just a few clicks"
                    }
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

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

              {/* Image Upload Section */}
              <div className="input-group">
                <label className="input-label">
                  {t.uploadImages}
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
                />
                
                {/* Display uploaded images */}
                {uploadedImages.length > 0 && (
                  <div className="image-grid">
                    {uploadedImages.map((image) => (
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
                        <p className="image-name">{image.name}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div style={{ display: "flex", gap: "1rem", justifyContent: "center" }}>
                <button 
                  onClick={fetchRecommendations}
                  disabled={isLoading}
                  className="btn"
                >
                  {isLoading ? "🔄" : "🔍"} {t.showRecommendations}
                </button>
                {(recommendations.length > 0 || inputText || uploadedImages.length > 0) && (
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
                      ⭐ {hotel.rating}
                    </div>
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