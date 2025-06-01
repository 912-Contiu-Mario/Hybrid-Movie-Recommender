"use client"

import { useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { searchExistingMovies } from "../services/api"
import MovieCard from "../components/MovieCard"
import LoadingSpinner from "../components/LoadingSpinner"

interface Movie {
  id: number           // Internal database ID
  tmdbId: number       // TMDB ID
  title: string
  poster_path: string
  release_date: string
  overview: string
}

const Search = () => {
  const [searchParams] = useSearchParams()
  const query = searchParams.get("q") || ""
  const [movies, setMovies] = useState<Movie[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    const fetchMovies = async () => {
      if (!query) return

      try {
        setLoading(true)
        setError("")
        const results = await searchExistingMovies(query)
        setMovies(results)
      } catch (err) {
        console.error("Search failed:", err)
        setError("Failed to search movies. Please try again.")
      } finally {
        setLoading(false)
      }
    }

    fetchMovies()
  }, [query])

  if (loading) {
    return <LoadingSpinner />
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">{query ? `Search results for "${query}"` : "Search Movies"}</h1>
        {movies.length > 0 && <p className="text-gray-400 mt-2">Found {movies.length} results in your database</p>}
      </div>

      {error && <div className="text-center text-red-500 py-8">{error}</div>}

      {!loading && !error && movies.length === 0 && (
        <div className="text-center text-gray-400 py-12">
          {query 
            ? "No movies found in your database. Try a different search term." 
            : "Enter a search term to find movies."
          }
        </div>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
        {movies.map((movie) => (
          <MovieCard
            key={movie.id}
            id={movie.id}
            tmdbId={movie.tmdbId}
            title={movie.title}
            posterPath={movie.poster_path}
            year={movie.release_date ? movie.release_date.split("-")[0] : undefined}
            overview={movie.overview}
          />
        ))}
      </div>
    </div>
  )
}

export default Search
