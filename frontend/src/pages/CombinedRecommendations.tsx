"use client"

import React, { useState, useEffect } from "react"
import { searchExistingMovies, getTMDBDetails, getCombinedRecommendations } from "../services/api"
import MovieCard from "../components/MovieCard"
import LoadingSpinner from "../components/LoadingSpinner"

interface Movie {
  id: number
  tmdbId: number
  title: string
  posterPath?: string
  releaseDate?: string
  overview?: string
  score?: number
  year?: string
}

const CombinedRecommendations = () => {
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<Movie[]>([])
  const [selectedMovies, setSelectedMovies] = useState<Movie[]>([])
  const [recommendations, setRecommendations] = useState<Movie[]>([])
  const [loading, setLoading] = useState(false)
  const [searchLoading, setSearchLoading] = useState(false)
  const [recLoading, setRecLoading] = useState(false)
  const [error, setError] = useState("")

  // Load saved data from localStorage on component mount
  useEffect(() => {
    const savedSelectedMovies = localStorage.getItem('magicSelectedMovies')
    const savedRecommendations = localStorage.getItem('magicRecommendations')
    
    if (savedSelectedMovies) {
      try {
        setSelectedMovies(JSON.parse(savedSelectedMovies))
      } catch (err) {
        console.error('Failed to parse saved selected movies:', err)
      }
    }
    
    if (savedRecommendations) {
      try {
        setRecommendations(JSON.parse(savedRecommendations))
      } catch (err) {
        console.error('Failed to parse saved recommendations:', err)
      }
    }
  }, [])

  // Save selected movies to localStorage when they change
  useEffect(() => {
    localStorage.setItem('magicSelectedMovies', JSON.stringify(selectedMovies))
  }, [selectedMovies])

  // Save recommendations to localStorage when they change
  useEffect(() => {
    localStorage.setItem('magicRecommendations', JSON.stringify(recommendations))
  }, [recommendations])

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!searchQuery.trim()) return

    try {
      setSearchLoading(true)
      setError("")
      const results = await searchExistingMovies(searchQuery)
      // Get TMDB details for each movie
      const enhancedResults = await Promise.all(
        results.map(async (movie: { id: number; tmdbId: number; title: string }) => {
          try {
            const tmdbData = await getTMDBDetails(movie.tmdbId)
            return {
              ...movie,
              posterPath: tmdbData.poster_path,
              releaseDate: tmdbData.release_date,
              overview: tmdbData.overview,
              year: tmdbData.release_date ? tmdbData.release_date.split("-")[0] : undefined,
            }
          } catch (err) {
            console.error(`Failed to fetch TMDB data for movie ${movie.tmdbId}:`, err)
            return movie
          }
        })
      )
      setSearchResults(enhancedResults)
    } catch (err) {
      console.error("Search failed:", err)
      setError("Failed to search movies. Please try again.")
    } finally {
      setSearchLoading(false)
    }
  }

  const handleAddMovie = (movie: Movie) => {
    // Check if movie is already in the selected list
    if (!selectedMovies.some(m => m.id === movie.id)) {
      setSelectedMovies([...selectedMovies, movie])
    }
    // Clear search results and query
    setSearchResults([])
    setSearchQuery("")
  }

  const handleRemoveMovie = (movieId: number) => {
    setSelectedMovies(selectedMovies.filter(movie => movie.id !== movieId))
  }

  const handleGetRecommendations = async () => {
    if (selectedMovies.length === 0) {
      setError("Please select at least one movie first")
      return
    }

    try {
      setRecLoading(true)
      setError("")
      
      // Get movie IDs from selected movies
      const movieIds = selectedMovies.map(movie => movie.id)
      
      // Get combined recommendations
      const recommendationsData = await getCombinedRecommendations(movieIds, 20)
      
      // Enhance recommendations with TMDB data
      const enhancedRecommendations = await Promise.all(
        recommendationsData.map(async (movie: any) => {
          try {
            const tmdbData = await getTMDBDetails(movie.tmdb_id)
            return {
              id: movie.id,
              tmdbId: movie.tmdb_id,
              title: movie.title,
              score: movie.score,
              posterPath: tmdbData.poster_path,
              releaseDate: tmdbData.release_date,
              overview: tmdbData.overview,
              year: tmdbData.release_date ? tmdbData.release_date.split("-")[0] : undefined,
            }
          } catch (err) {
            console.error(`Failed to fetch TMDB data for movie ${movie.tmdb_id}:`, err)
            return {
              id: movie.id,
              tmdbId: movie.tmdb_id,
              title: movie.title,
              score: movie.score,
            }
          }
        })
      )
      
      setRecommendations(enhancedRecommendations)
    } catch (err) {
      console.error("Failed to get recommendations:", err)
      setError("Failed to get recommendations. Please try again.")
    } finally {
      setRecLoading(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-12 text-center">
        <h1 className="text-4xl font-bold text-white mb-2">Magic Recommender</h1>
        <p className="text-gray-400">Select multiple movies and watch the magic happen!</p>
      </div>

      {/* Search Section */}
      <div className="max-w-2xl mx-auto mb-12">
        <form onSubmit={handleSearch} className="relative">
          <input
            type="text"
            placeholder="Search for movies to add..."
            className="w-full bg-gray-800 text-white px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-600"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button
            type="submit"
            disabled={searchLoading}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 bg-red-600 text-white px-4 py-1 rounded-full hover:bg-red-700 disabled:opacity-50"
          >
            {searchLoading ? "Searching..." : "Search"}
          </button>
        </form>

        {/* Search Results */}
        {searchResults.length > 0 && (
          <div className="mt-4 bg-gray-900 rounded-lg p-4 max-h-96 overflow-y-auto">
            <h2 className="text-lg font-semibold text-white mb-3">Search Results</h2>
            <div className="space-y-3">
              {searchResults.map((movie) => (
                <div
                  key={movie.id}
                  className="flex items-center gap-4 p-2 hover:bg-gray-800 rounded cursor-pointer"
                  onClick={() => handleAddMovie(movie)}
                >
                  <img
                    src={
                      movie.posterPath
                        ? `https://image.tmdb.org/t/p/w92${movie.posterPath}`
                        : "/placeholder_poster.jpg"
                    }
                    alt={movie.title}
                    className="w-12 h-18 object-cover rounded"
                    onError={(e) => {
                      e.currentTarget.src = "/placeholder_poster.jpg"
                    }}
                  />
                  <div className="flex-1">
                    <h3 className="font-medium text-white">{movie.title}</h3>
                    <p className="text-sm text-gray-400">
                      {movie.year || "Unknown year"}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleAddMovie(movie)
                    }}
                    className="bg-red-600 text-white px-3 py-1 rounded-full text-sm hover:bg-red-700"
                  >
                    Add
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {searchResults.length === 0 && searchQuery && !searchLoading && (
          <div className="mt-4 text-center text-gray-400">
            No movies found in your database. Try a different search term.
          </div>
        )}
      </div>

      {/* Selected Movies Section */}
      {selectedMovies.length > 0 && (
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-white mb-6">Selected Movies</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {selectedMovies.map((movie) => (
              <div key={movie.id} className="relative">
                <div className="bg-gray-900 rounded-lg overflow-hidden">
                  <img
                    src={
                      movie.posterPath
                        ? `https://image.tmdb.org/t/p/w300${movie.posterPath}`
                        : "/placeholder_poster.jpg"
                    }
                    alt={movie.title}
                    className="w-full h-auto object-cover"
                    onError={(e) => {
                      e.currentTarget.src = "/placeholder_poster.jpg"
                    }}
                  />
                  <div className="p-2">
                    <h3 className="font-medium text-white text-sm truncate">{movie.title}</h3>
                    <p className="text-xs text-gray-400">{movie.year || "Unknown"}</p>
                  </div>
                </div>
                <button
                  onClick={() => handleRemoveMovie(movie.id)}
                  className="absolute top-2 right-2 bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center hover:bg-red-700"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          <div className="mt-6 text-center">
            <button
              onClick={handleGetRecommendations}
              disabled={recLoading || selectedMovies.length === 0}
              className="bg-red-600 text-white px-6 py-3 rounded-lg text-lg hover:bg-red-700 disabled:opacity-50"
            >
              {recLoading ? <LoadingSpinner size="sm" /> : "Create Magic ✨"}
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && <div className="text-center text-red-500 py-4 mb-8">{error}</div>}

      {/* Recommendations Section */}
      {recLoading ? (
        <div className="py-12 text-center">
          <LoadingSpinner />
          <p className="text-gray-400 mt-4">Creating magic just for you...</p>
        </div>
      ) : recommendations.length > 0 ? (
        <div>
          <h2 className="text-2xl font-bold text-white mb-6">Your Magical Recommendations ✨</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
            {recommendations.map((movie) => (
              <MovieCard
                key={movie.id}
                id={movie.id}
                tmdbId={movie.tmdbId}
                title={movie.title}
                posterPath={movie.posterPath}
                score={movie.score}
                year={movie.year}
                overview={movie.overview}
              />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default CombinedRecommendations 