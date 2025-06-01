"use client"

import { useEffect, useState } from "react"
import { getUserRatings, getMovieByInternalId, deleteRating } from "../services/api"
import MovieCard from "../components/MovieCard"
import LoadingSpinner from "../components/LoadingSpinner"
import ReadOnlyStarRating from "../components/ReadOnlyStarRating"

interface RatedMovie {
  id: number
  tmdbId: number
  title: string
  posterPath?: string
  releaseDate?: string
  overview?: string
  rating: number
  year?: string
}

const RatedMovies = () => {
  const [ratedMovies, setRatedMovies] = useState<RatedMovie[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [sortOrder, setSortOrder] = useState<"highest" | "lowest" | "newest" | "oldest">("highest")

  useEffect(() => {
    fetchRatedMovies()
  }, [])

  const fetchRatedMovies = async () => {
    try {
      setLoading(true)
      const ratings = await getUserRatings()

      // Get full movie details for each rated movie
      const moviesWithDetails = await Promise.all(
        ratings.map(async (rating: any) => {
          try {
            const movieData = await getMovieByInternalId(rating.movie_id)
            return {
              id: movieData.id,
              tmdbId: movieData.tmdbId,
              title: movieData.title,
              posterPath: movieData.poster_path,
              releaseDate: movieData.release_date,
              overview: movieData.overview,
              rating: rating.rating,
              year: movieData.release_date ? new Date(movieData.release_date).getFullYear() : undefined,
            }
          } catch (err) {
            console.error(`Failed to fetch movie details for ID ${rating.movie_id}:`, err)
            return null
          }
        })
      )

      // Filter out any failed fetches and sort by rating
      const validMovies = moviesWithDetails.filter((movie): movie is RatedMovie => movie !== null)
      setRatedMovies(validMovies)
      sortMovies(validMovies, sortOrder)
    } catch (err) {
      console.error("Failed to fetch rated movies:", err)
      setError("Failed to load your rated movies. Please try again later.")
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteRating = async (movieId: number) => {
    try {
      await deleteRating(movieId)
      setRatedMovies((prev) => prev.filter((movie) => movie.id !== movieId))
    } catch (err) {
      console.error("Failed to delete rating:", err)
      alert("Failed to delete rating. Please try again.")
    }
  }

  const sortMovies = (movies: RatedMovie[], order: typeof sortOrder) => {
    let sortedMovies = [...movies]
    switch (order) {
      case "highest":
        sortedMovies.sort((a, b) => b.rating - a.rating)
        break
      case "lowest":
        sortedMovies.sort((a, b) => a.rating - b.rating)
        break
      case "newest":
        sortedMovies.sort((a, b) => {
          if (!a.releaseDate || !b.releaseDate) return 0
          return new Date(b.releaseDate).getTime() - new Date(a.releaseDate).getTime()
        })
        break
      case "oldest":
        sortedMovies.sort((a, b) => {
          if (!a.releaseDate || !b.releaseDate) return 0
          return new Date(a.releaseDate).getTime() - new Date(b.releaseDate).getTime()
        })
        break
    }
    setRatedMovies(sortedMovies)
  }

  const handleSortChange = (newOrder: typeof sortOrder) => {
    setSortOrder(newOrder)
    sortMovies(ratedMovies, newOrder)
  }

  if (loading) {
    return <LoadingSpinner />
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-white text-center">
          <p className="text-xl">{error}</p>
          <button
            onClick={fetchRatedMovies}
            className="mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-12">
        <h1 className="text-4xl font-bold text-white mb-4">Your Rated Movies</h1>
        <p className="text-gray-400 mb-8">
          {ratedMovies.length} {ratedMovies.length === 1 ? "movie" : "movies"} rated
        </p>

        {/* Sort Controls */}
        <div className="flex flex-wrap gap-4 mb-8">
          <button
            onClick={() => handleSortChange("highest")}
            className={`px-4 py-2 rounded-full text-sm ${
              sortOrder === "highest"
                ? "bg-red-600 text-white"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            Highest Rated
          </button>
          <button
            onClick={() => handleSortChange("lowest")}
            className={`px-4 py-2 rounded-full text-sm ${
              sortOrder === "lowest"
                ? "bg-red-600 text-white"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            Lowest Rated
          </button>
          <button
            onClick={() => handleSortChange("newest")}
            className={`px-4 py-2 rounded-full text-sm ${
              sortOrder === "newest"
                ? "bg-red-600 text-white"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            Release Date (Newest)
          </button>
          <button
            onClick={() => handleSortChange("oldest")}
            className={`px-4 py-2 rounded-full text-sm ${
              sortOrder === "oldest"
                ? "bg-red-600 text-white"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            Release Date (Oldest)
          </button>
        </div>
      </div>

      {ratedMovies.length === 0 ? (
        <div className="text-center text-gray-400 py-12">
          <p className="text-xl mb-4">You haven't rated any movies yet</p>
          <p>Rate movies to keep track of what you've watched and get better recommendations</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-8">
          {ratedMovies.map((movie) => (
            <div key={movie.id} className="relative">
              <MovieCard
                id={movie.id}
                tmdbId={movie.tmdbId}
                title={movie.title}
                posterPath={movie.posterPath}
                year={movie.year}
                overview={movie.overview}
              />
              <div className="mt-2">
                <div className="flex items-center justify-between">
                  <ReadOnlyStarRating rating={movie.rating} size="sm" />
                  <button
                    onClick={() => handleDeleteRating(movie.id)}
                    className="text-red-500 hover:text-red-400 text-sm"
                  >
                    Remove
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default RatedMovies 