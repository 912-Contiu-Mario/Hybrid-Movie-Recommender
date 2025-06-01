"use client"

import type React from "react"

import { useState } from "react"
import { searchExistingMovies, getSimilarMovies, getTMDBDetails } from "../services/api"
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

const SimilarMovies = () => {
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<Movie[]>([])
  const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null)
  const [similarMovies, setSimilarMovies] = useState<Movie[]>([])
  const [loading, setLoading] = useState(false)
  const [searchLoading, setSearchLoading] = useState(false)
  const [error, setError] = useState("")

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

  const handleSelectMovie = async (movie: Movie) => {
    setSelectedMovie(movie)
    setSearchResults([])
    setSearchQuery("")

    try {
      setLoading(true)
      setError("")
      const similarMoviesData = await getSimilarMovies(movie.id, 20)

      // Enhance similar movies with TMDB data
      const enhancedMovies = await Promise.all(
        similarMoviesData.map(async (movie: any) => {
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
        }),
      )

      setSimilarMovies(enhancedMovies)
    } catch (err) {
      console.error("Failed to get similar movies:", err)
      setError("Failed to get similar movies. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-12 text-center">
        <h1 className="text-4xl font-bold text-white mb-2">Find Similar Movies</h1>
        <p className="text-gray-400">Search for a movie to find similar recommendations</p>
      </div>

      <div className="max-w-2xl mx-auto mb-12">
        <form onSubmit={handleSearch} className="relative">
          <input
            type="text"
            placeholder="Search for a movie..."
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
                  onClick={() => handleSelectMovie(movie)}
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
                  <div>
                    <h3 className="font-medium text-white">{movie.title}</h3>
                    <p className="text-sm text-gray-400">
                      {movie.releaseDate ? new Date(movie.releaseDate).getFullYear() : "Unknown"}
                    </p>
                  </div>
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

      {/* Selected Movie */}
      {selectedMovie && (
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-white mb-6">Similar to</h2>
          <div className="flex items-center gap-6 bg-gray-900 p-6 rounded-lg">
            <img
              src={
                selectedMovie.posterPath
                  ? `https://image.tmdb.org/t/p/w185${selectedMovie.posterPath}`
                  : "/placeholder_poster.jpg"
              }
              alt={selectedMovie.title}
              className="w-32 rounded-lg"
              onError={(e) => {
                e.currentTarget.src = "/placeholder_poster.jpg"
              }}
            />
            <div>
              <h3 className="text-2xl font-bold text-white">{selectedMovie.title}</h3>
              <p className="text-gray-400">
                {selectedMovie.releaseDate ? new Date(selectedMovie.releaseDate).getFullYear() : "Unknown"}
              </p>
              <p className="text-gray-300 mt-2 line-clamp-3">{selectedMovie.overview}</p>
            </div>
          </div>
        </div>
      )}

      {/* Similar Movies */}
      {loading ? (
        <LoadingSpinner />
      ) : error ? (
        <div className="text-center text-red-500 py-8">{error}</div>
      ) : similarMovies.length > 0 ? (
        <div>
          <h2 className="text-2xl font-bold text-white mb-6">Similar Movies</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
            {similarMovies.map((movie) => (
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
      ) : selectedMovie ? (
        <div className="text-center text-gray-400 py-8">No similar movies found. Try another movie.</div>
      ) : null}
    </div>
  )
}

export default SimilarMovies
