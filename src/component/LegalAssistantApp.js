import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Plus,
  MessageSquare,
  Trash2,
  FileText,
  Star,
  Settings,
  Scale,
  Calendar,
  Filter,
  BookOpen,
  Search,
  AlertTriangle,
  Mic,
  MicOff,
  Volume2,
  VolumeX
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Voice Interface Component
const VoiceInterface = ({ onSpeechResult, messageToSpeak, isProcessing }) => {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const recognitionRef = useRef(null);
  const utteranceRef = useRef(null);
  const previousMessageRef = useRef('');
  
  // Initialize speech recognition and synthesis on component mount
  useEffect(() => {
    // Check for browser support
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      
      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0])
          .map(result => result.transcript)
          .join('');
          
        if (event.results[0].isFinal) {
          onSpeechResult(transcript);
          stopListening();
        }
      };
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        stopListening();
      };
      
      setSpeechEnabled(true);
    }
    
    if ('speechSynthesis' in window) {
      setVoiceSupported(true);
    }
    
    // Cleanup on unmount
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
  }, [onSpeechResult]);
  
  // Handle speaking the bot's messages
  useEffect(() => {
    if (!voiceSupported || !messageToSpeak || messageToSpeak === previousMessageRef.current || isProcessing) {
      return;
    }
    
    previousMessageRef.current = messageToSpeak;
    
    // Check if it's a bot message (filtering out disclaimers and citations)
    const mainContent = messageToSpeak.split('Citations:')[0].split('Disclaimer:')[0];
    
    if (isSpeaking) {
      window.speechSynthesis.cancel();
    }
    
    utteranceRef.current = new SpeechSynthesisUtterance(mainContent);
    
    // Get voices and set a professional-sounding voice if available
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(voice => 
      voice.name.includes('Daniel') || // Professional-sounding male voice
      voice.name.includes('Samantha') || // Professional-sounding female voice
      voice.name.includes('Google UK English') ||
      voice.name.includes('Microsoft')
    );
    
    if (preferredVoice) {
      utteranceRef.current.voice = preferredVoice;
    }
    
    utteranceRef.current.rate = 1.0;
    utteranceRef.current.pitch = 1.0;
    
    utteranceRef.current.onstart = () => setIsSpeaking(true);
    utteranceRef.current.onend = () => setIsSpeaking(false);
    utteranceRef.current.onerror = (event) => {
      console.error('Speech synthesis error', event);
      setIsSpeaking(false);
    };
    
    window.speechSynthesis.speak(utteranceRef.current);
  }, [messageToSpeak, voiceSupported, isProcessing, isSpeaking]);
  
  // Start listening for speech input
  const startListening = () => {
    if (!speechEnabled || isListening) return;
    
    try {
      recognitionRef.current.start();
      setIsListening(true);
    } catch (error) {
      console.error('Error starting speech recognition:', error);
    }
  };
  
  // Stop listening for speech input
  const stopListening = () => {
    if (!speechEnabled || !isListening) return;
    
    try {
      recognitionRef.current.stop();
      setIsListening(false);
    } catch (error) {
      console.error('Error stopping speech recognition:', error);
    }
  };
  
  // Toggle speech output
  const toggleSpeaking = () => {
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    } else if (messageToSpeak && !isProcessing) {
      // Restart speech for the last message
      previousMessageRef.current = '';  // Reset to allow re-speaking
      window.speechSynthesis.cancel();
    }
  };
  
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={isListening ? stopListening : startListening}
        disabled={!speechEnabled}
        className={`p-2 rounded-full ${
          !speechEnabled 
            ? "bg-slate-800/30 text-slate-600 cursor-not-allowed" 
            : isListening
              ? "bg-red-600 text-white animate-pulse"
              : "bg-slate-800 text-slate-300 hover:bg-slate-700"
        }`}
        title={speechEnabled ? (isListening ? "Stop listening" : "Start voice input") : "Speech recognition not supported"}
      >
        {isListening ? <MicOff size={18} /> : <Mic size={18} />}
      </button>
      
      <button
        onClick={toggleSpeaking}
        disabled={!voiceSupported || isProcessing}
        className={`p-2 rounded-full ${
          !voiceSupported 
            ? "bg-slate-800/30 text-slate-600 cursor-not-allowed" 
            : isSpeaking
              ? "bg-indigo-600 text-white"
              : "bg-slate-800 text-slate-300 hover:bg-slate-700"
        }`}
        title={voiceSupported ? (isSpeaking ? "Stop speaking" : "Read response aloud") : "Speech synthesis not supported"}
      >
        {isSpeaking ? <VolumeX size={18} /> : <Volume2 size={18} />}
      </button>
    </div>
  );
};

const LegalAssistantApp = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [consultationHistory, setConsultationHistory] = useState([]);
  const [currentConsultationId, setCurrentConsultationId] = useState(null);
  const [error, setError] = useState(null);
  const [jurisdiction, setJurisdiction] = useState("US Federal");
  const [caseDetails, setCaseDetails] = useState({});
  const [showCaseDetailsForm, setShowCaseDetailsForm] = useState(false);
  const chatboxRef = useRef(null);
  
  // New state for voice features
  const [latestBotMessage, setLatestBotMessage] = useState("");

  useEffect(() => {
    if (chatboxRef.current) {
      chatboxRef.current.scrollTop = chatboxRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const newUserMessage = { sender: "user", text: inputMessage };
    setMessages((prev) => [...prev, newUserMessage]);
    setInputMessage("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/legal/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          query: inputMessage,
          jurisdiction: jurisdiction,
          case_details: Object.keys(caseDetails).length > 0 ? caseDetails : null
        }),
        credentials: "omit",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.status === "error") {
        throw new Error(data.error);
      }

      // Format bot message to include legal disclaimer
      const botMessage = { 
        sender: "bot", 
        text: data.analysis,
        disclaimer: data.disclaimer,
        citations: data.citations || []
      };
      
      setMessages((prev) => [...prev, botMessage]);
      // Set the latest message for speech synthesis
      setLatestBotMessage(data.analysis);

      // Update consultation history
      const updatedMessages = [...messages, newUserMessage, botMessage];
      if (currentConsultationId) {
        setConsultationHistory((prev) =>
          prev.map((chat) =>
            chat.id === currentConsultationId
              ? { ...chat, messages: updatedMessages }
              : chat
          )
        );
      } else {
        const newConsultationId = Date.now();
        const newConsultation = {
          id: newConsultationId,
          title: inputMessage.slice(0, 30),
          messages: updatedMessages,
          jurisdiction: jurisdiction,
          timestamp: new Date().toLocaleString(),
          caseDetails: {...caseDetails}
        };
        setConsultationHistory((prev) => [...prev, newConsultation]);
        setCurrentConsultationId(newConsultationId);
      }
    } catch (error) {
      console.error("Error:", error);
      setError(error.message);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "I'm sorry, I'm having trouble processing your legal query. Please try again later.",
          disclaimer: "This error response is not legal advice."
        },
      ]);
      // Set error message for speech as well
      setLatestBotMessage("I'm sorry, I'm having trouble processing your legal query. Please try again later.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };
  
  // Voice input handler
  const handleSpeechInput = (transcript) => {
    setInputMessage(transcript);
    // Uncomment to auto-send after voice input
    setTimeout(() => handleSend(), 500);
  };

  const startNewConsultation = () => {
    setCurrentConsultationId(null);
    setMessages([]);
    setCaseDetails({});
    setShowCaseDetailsForm(false);
  };

  const loadConsultation = (consultationId) => {
    const consultation = consultationHistory.find((c) => c.id === consultationId);
    if (consultation) {
      setCurrentConsultationId(consultationId);
      setMessages(consultation.messages);
      setJurisdiction(consultation.jurisdiction || "US Federal");
      setCaseDetails(consultation.caseDetails || {});
    }
  };

  const deleteConsultation = (consultationId, e) => {
    e.stopPropagation();
    setConsultationHistory((prev) => prev.filter((chat) => chat.id !== consultationId));
    if (currentConsultationId === consultationId) {
      setCurrentConsultationId(null);
      setMessages([]);
      setCaseDetails({});
    }
  };

  const handleCaseDetailChange = (key, value) => {
    setCaseDetails(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const addNewCaseDetailField = () => {
    setCaseDetails(prev => ({
      ...prev,
      [`detail_${Object.keys(caseDetails).length + 1}`]: ""
    }));
  };

  const messageVariants = {
    initial: (custom) => ({
      opacity: 0,
      x: custom === "bot" ? -100 : 100,
      y: 50,
      scale: 0.8,
      rotateX: 45,
    }),
    animate: {
      opacity: 1,
      x: 0,
      y: 0,
      scale: 1,
      rotateX: 0,
      transition: {
        type: "spring",
        stiffness: 200,
        damping: 20,
        mass: 0.8,
      },
    },
    exit: {
      opacity: 0,
      scale: 0.5,
      y: 100,
      transition: { 
        duration: 0.3,
        ease: "easeOut" 
      },
    },
    hover: {
      scale: 1.02,
      transition: {
        duration: 0.2
      }
    }
  };
  
  const [searchTerm, setSearchTerm] = useState("");
  const [favorites, setFavorites] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Categories for consultations
  const categories = [
    { key: 1, id: "all", name: "All Consultations", icon: MessageSquare, link: "/" },
    { key: 2, id: "favorites", name: "Favorites", icon: Star, link: "/" },
    { key: 3, id: "resources", name: "Legal Resources", icon: BookOpen, link: "/" },
    { key: 4, id: "precedents", name: "Case Precedents", icon: Scale, link: "/" },
    { key: 5, id: "calendar", name: "Case Calendar", icon: Calendar, link: "/" },
  ];

  // Jurisdiction options
  const jurisdictionOptions = [
    "US Federal",
    "Indian Penal Code",
    "Indian Contract Act",
    "California",
    "New York",
    "Texas",
    "Florida",
    "UK",
    "EU",
    "Canada",
    "Australia"
  ];

  const toggleFavorite = (consultationId, e) => {
    e.stopPropagation();
    setFavorites(prev => 
      prev.includes(consultationId) 
        ? prev.filter(id => id !== consultationId)
        : [...prev, consultationId]
    );
  };

  const filteredConsultations = consultationHistory.filter(consultation => {
    const matchesSearch = consultation.title.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === "all" || 
      (selectedCategory === "favorites" && favorites.includes(consultation.id));
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="flex h-screen bg-gradient-to-b from-slate-900 to-slate-950 text-slate-100">
      {/* Enhanced Sidebar */}
      <div
        className={`${
          isSidebarCollapsed ? "w-20" : "w-64"
        } border-r border-slate-700/20 p-4 transition-all duration-300 ease-in-out`}
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between mb-6">
          {!isSidebarCollapsed && (
            <h2 className="text-xl font-bold text-slate-200">
              Case Files
            </h2>
          )}
          <button
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            className="p-2 rounded-lg bg-slate-800/50 hover:bg-slate-800"
          >
            <Filter size={16} />
          </button>
        </div>

        {/* Search Bar */}
        {!isSidebarCollapsed && (
          <div className="relative mb-4">
            <Search size={16} className="absolute left-3 top-3 text-slate-400" />
            <input
              type="text"
              placeholder="Search cases..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-800/30 border border-slate-700/30 rounded-lg py-2 pl-10 pr-4 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
        )}

        {/* Action Buttons */}
        <button
          onClick={startNewConsultation}
          className={`w-full bg-gradient-to-r from-indigo-600 to-indigo-800 text-slate-100 rounded-lg p-3 flex items-center ${
            isSidebarCollapsed ? "justify-center" : "justify-between"
          } gap-2 shadow-lg mb-4 hover:from-indigo-500 hover:to-indigo-700`}
        >
          <Plus size={20} />
          {!isSidebarCollapsed && <span className="font-semibold">New Consultation</span>}
        </button>

        {/* Categories */}
        <div className="mb-6 space-y-2 overflow-auto">
          {categories.map((category) => (
            <button
              key={category.key}
              onClick={() => setSelectedCategory(category.id)}
              className={`w-full p-2 rounded-lg flex items-center ${
                isSidebarCollapsed ? "justify-center" : "justify-start"
              } gap-3 ${
                selectedCategory === category.id
                  ? "bg-slate-800 text-white"
                  : "text-slate-400 hover:bg-slate-800/50"
              }`}
            >
              <category.icon size={16} />
              {!isSidebarCollapsed && <span>{category.name}</span>}
            </button>
          ))}
        </div>

        {/* Consultation List */}
        <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
          {filteredConsultations.map((consultation, index) => (
            <div
              key={consultation.id}
              className={`flex items-center justify-between p-3 rounded-lg cursor-pointer ${
                currentConsultationId === consultation.id
                  ? "bg-slate-800/70 border border-slate-700"
                  : "hover:bg-slate-800/40 border border-transparent"
              }`}
              onClick={() => loadConsultation(consultation.id)}
            >
              <div className="flex items-center space-x-3 text-slate-300 min-w-0">
                <FileText size={16} className="text-slate-500 flex-shrink-0" />
                {!isSidebarCollapsed && (
                  <div className="overflow-hidden">
                    <span className="truncate font-medium block">{consultation.title}</span>
                    <span className="text-xs text-slate-500">{consultation.jurisdiction} • {consultation.timestamp}</span>
                  </div>
                )}
              </div>
              {!isSidebarCollapsed && (
                <div className="flex items-center space-x-2">
                  <button
                    onClick={(e) => toggleFavorite(consultation.id, e)}
                    className={`text-slate-400 hover:text-yellow-500 ${
                      favorites.includes(consultation.id) ? "text-yellow-500" : ""
                    }`}
                  >
                    <Star size={16} />
                  </button>
                  <button
                    onClick={(e) => deleteConsultation(consultation.id, e)}
                    className="text-slate-400 hover:text-red-500"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Settings Button */}
        <button
          className={`bg-slate-800/50 text-slate-400 rounded-lg p-3 flex items-center ${
            isSidebarCollapsed ? "justify-center" : "justify-between"
          } gap-2 mt-auto absolute bottom-4 left-4`}
        >
          <Settings size={20} />
          {!isSidebarCollapsed && <span>Settings</span>}
        </button>
      </div>

      {/* Main Consultation Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b border-slate-700/20 p-4">
          <div className="flex items-center justify-center space-x-4">
            <div className="w-12 h-12 rounded-full bg-gradient-to-r from-indigo-700 to-indigo-900 flex items-center justify-center">
              <Scale size={24} className="text-slate-200" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-300 to-slate-400 text-transparent bg-clip-text">
              LegalAssist AI
            </h1>
          </div>
          
          {/* Jurisdiction selector */}
          <div className="mt-4 flex flex-col md:flex-row items-center justify-center gap-4">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-slate-400">Jurisdiction:</span>
              <select 
                value={jurisdiction} 
                
                  onChange={(e) => setJurisdiction(e.target.value)}
                  className="bg-slate-800/50 border border-slate-700/30 rounded px-3 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  {jurisdictionOptions.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>
  
              <button 
                onClick={() => setShowCaseDetailsForm(!showCaseDetailsForm)}
                className="text-sm bg-slate-800/50 hover:bg-slate-800 text-slate-300 flex items-center gap-2 px-4 py-1 rounded border border-slate-700/30"
              >
                {showCaseDetailsForm ? "Hide" : "Add"} Case Details <FileText size={14} />
              </button>
            </div>
          </div>
  
          {/* Case Details Form */}
          <AnimatePresence>
            {showCaseDetailsForm && (
              <motion.div 
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="border-b border-slate-700/20 bg-slate-900/50 overflow-hidden"
              >
                <div className="p-4 max-w-3xl mx-auto">
                  <h3 className="text-lg font-medium mb-3 text-slate-300">Case Details</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    {Object.keys(caseDetails).map((key) => (
                      <div key={key} className="flex flex-col">
                        <div className="flex justify-between items-center mb-1">
                          <input
                            type="text"
                            value={key}
                            onChange={(e) => {
                              const newCaseDetails = {...caseDetails};
                              const value = caseDetails[key];
                              delete newCaseDetails[key];
                              newCaseDetails[e.target.value] = value;
                              setCaseDetails(newCaseDetails);
                            }}
                            placeholder="Field name"
                            className="text-sm font-medium text-slate-300 bg-transparent border-b border-slate-700/30 focus:outline-none focus:border-indigo-500 w-full"
                          />
                          <button 
                            onClick={() => {
                              const newCaseDetails = {...caseDetails};
                              delete newCaseDetails[key];
                              setCaseDetails(newCaseDetails);
                            }}
                            className="text-slate-500 hover:text-red-500 ml-2"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                        <textarea
                          value={caseDetails[key]}
                          onChange={(e) => handleCaseDetailChange(key, e.target.value)}
                          placeholder="Enter details..."
                          rows={2}
                          className="bg-slate-800/30 border border-slate-700/30 rounded p-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 resize-none"
                        />
                      </div>
                    ))}
                  </div>
                  
                  <button 
                    onClick={addNewCaseDetailField}
                    className="text-sm bg-slate-800/50 hover:bg-slate-800 text-slate-300 flex items-center gap-2 px-3 py-1 rounded border border-slate-700/30"
                  >
                    Add Field <Plus size={14} />
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
  
          {/* Messages */}
          <div ref={chatboxRef} className="flex-1 overflow-y-auto p-6 space-y-6">
            <AnimatePresence mode="popLayout">
              {messages.map((message, index) => (
                <motion.div
                  key={index}
                  custom={message.sender}
                  variants={messageVariants}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  className={`flex ${message.sender === "bot" ? "justify-start" : "justify-end"}`}
                >
                  <div
                    className={`max-w-[80%] p-4 rounded-2xl ${
                      message.sender === "bot"
                        ? "bg-slate-900/80 rounded-tl-none border border-slate-800"
                        : "bg-indigo-900/80 rounded-br-none"
                    }`}
                  >
                    <div className="flex items-start space-x-4">
                      {message.sender === "bot" && (
                        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-700 to-indigo-900 flex items-center justify-center">
                          <Scale size={16} className="text-slate-100" />
                        </div>
                      )}
                      <div className="flex-1 text-slate-100 break-words">
                        <div className="prose prose-invert prose-slate prose-sm">
                          {message.text}
                          
                          {/* Render citations if available */}
                          {message.citations && message.citations.length > 0 && (
                            <div className="mt-3 text-xs border-t border-slate-700/50 pt-2">
                              <p className="font-medium text-slate-400">Citations:</p>
                              <ul className="list-disc pl-4 mt-1 text-slate-400">
                                {message.citations.map((citation, i) => (
                                  <li key={i}>{citation}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {/* Render disclaimer if available */}
                          {message.disclaimer && (
                            <div className="mt-3 text-xs border-t border-slate-700/50 pt-2 flex items-start gap-1 text-amber-500">
                              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
                              <p>{message.disclaimer}</p>
                            </div>
                          )}
                        </div>
                      </div>
                      {message.sender === "user" && (
                        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-slate-700 to-slate-800 flex items-center justify-center">
                          <span className="text-xs font-medium">You</span>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
  
            {/* Loading animation */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-[80%] p-4 bg-slate-900/80 rounded-2xl rounded-tl-none border border-slate-800">
                  <div className="flex items-start space-x-4">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-700 to-indigo-900 flex items-center justify-center animate-pulse">
                      <Scale size={16} className="text-slate-100" />
                    </div>
                    <div className="flex space-x-2">
                      {[0, 1, 2].map((i) => (
                        <div
                          key={i}
                          className="w-3 h-3 bg-slate-600 rounded-full"
                          style={{
                            animation: `bounce 1.4s ease-in-out ${i * 0.2}s infinite both`
                          }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
  
            {/* Welcome message when no messages */}
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center px-4 py-10">
                <Scale size={64} className="text-indigo-500 mb-6" />
                <h2 className="text-2xl font-semibold text-slate-300 mb-2">Welcome to LegalAssist AI</h2>
                <p className="text-slate-400 max-w-lg mb-6">
                  Your AI-powered legal research assistant. Ask any legal question or describe a legal scenario to receive analysis and guidance based on relevant legal frameworks.
                </p>
                <div className="bg-slate-800/50 border border-slate-700/30 rounded-lg p-4 max-w-lg w-full">
                  <h3 className="text-slate-300 font-medium mb-2">Try asking about:</h3>
                  <ul className="text-slate-400 text-sm space-y-2">
                    <li>"What are the requirements for forming a valid contract?"</li>
                    <li>"Explain the fair use doctrine in copyright law."</li>
                    <li>"What legal considerations should I keep in mind when starting an LLC?"</li>
                    <li>"Can you analyze the potential liability issues in this scenario..."</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
  
          {/* Input area */}
          <div className="border-t border-slate-700/20 p-6 bg-slate-950/40">
            <div className="max-w-3xl mx-auto relative">
              <div className="relative flex items-center">
                <textarea
                  rows={1}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter your legal question or describe a scenario..."
                  className="w-full bg-slate-900/50 text-white rounded-lg pl-6 pr-24 py-4 focus:outline-none focus:ring-2 focus:ring-indigo-500 border border-slate-800 min-h-[56px] resize-none"
                  disabled={isLoading}
                  style={{ height: 'auto', maxHeight: '120px' }}
                  onInput={e => {
                    e.target.style.height = 'auto';
                    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
                  }}
                />
                <div className="absolute right-2 flex items-center space-x-2">
                  <VoiceInterface 
                    onSpeechResult={handleSpeechInput}
                    messageToSpeak={latestBotMessage}
                    isProcessing={isLoading}
                  />
                  <button
                    onClick={handleSend}
                    disabled={isLoading || !inputMessage.trim()}
                    className={`p-2 rounded-lg ${
                      isLoading || !inputMessage.trim()
                        ? "bg-slate-800/50 text-slate-500 cursor-not-allowed"
                        : "bg-gradient-to-r from-indigo-600 to-indigo-800 text-white hover:shadow-lg"
                    }`}
                  >
                    <Send size={20} />
                  </button>
                </div>
              </div>
              <p className="text-xs text-slate-500 mt-2 ml-1">
                Not a substitute for professional legal advice. Results may vary based on jurisdiction.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  export default LegalAssistantApp;