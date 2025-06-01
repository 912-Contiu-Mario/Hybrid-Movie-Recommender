"use client"

import { useEffect, useState } from "react"
import { useParams } from "react-router-dom"
import { getMovieByInternalId, getMovieCredits, getUserRatings, rateMovie, deleteRating, getSimilarMovies, getTMDBDetails } from "../services/api"
import StarRating from "../components/StarRating"
import LoadingSpinner from "../components/LoadingSpinner"
import MovieCard from "../components/MovieCard"
import axios from "axios"

// Use axios directly since we don't have access to the API instances
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
const TMDB_API_KEY = import.meta.env.VITE_TMDB_API_KEY
const TMDB_BASE_URL = "https://api.themoviedb.org/3"

interface MovieDetailsData {
  id: number
  tmdbId?: number
  title: string
  overview: string
  poster_path: string
  backdrop_path: string
  release_date: string
  runtime: number
  vote_average: number
  genres: { id: number; name: string }[]
}

interface Credits {
  cast: {
    id: number
    name: string
    character: string
    profile_path: string | null
  }[]
  crew: {
    id: number
    name: string
    job: string
  }[]
}

interface UserRating {
  movie_id: number
  rating: number
  timestamp: string
}

interface SimilarMovie {
  id: number
  tmdbId: number
  title: string
  posterPath?: string
  score?: number
  year?: string
  overview?: string
}

const MovieDetails = () => {
  const { id } = useParams<{ id: string }>()
  const [movie, setMovie] = useState<MovieDetailsData | null>(null)
  const [credits, setCredits] = useState<Credits | null>(null)
  const [userRating, setUserRating] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [ratingLoading, setRatingLoading] = useState(false)
  const [similarMovies, setSimilarMovies] = useState<SimilarMovie[]>([])
  const [similarMoviesLoading, setSimilarMoviesLoading] = useState(false)

  useEffect(() => {
    const fetchMovieData = async () => {
      if (!id) return

      try {
        setLoading(true)
        setSimilarMoviesLoading(true)
        // Get the movie by internal ID
        const movieData = await getMovieByInternalId(Number(id))
        
        // Get additional data
        const [creditsData, ratingsData, similarMoviesData] = await Promise.all([
          getMovieCredits(movieData.tmdbId),
          getUserRatings(),
          getSimilarMovies(Number(id), 10)
        ])

        setMovie(movieData)
        setCredits(creditsData)

        // Find if user has rated this movie
        const userRatingObj = ratingsData.find((rating: UserRating) => rating.movie_id === Number(id))
        if (userRatingObj) {
          setUserRating(userRatingObj.rating)
        }

        // Process similar movies
        const enhancedSimilarMovies = await Promise.all(
          similarMoviesData.map(async (movie: any) => {
            try {
              const tmdbData = await getTMDBDetails(movie.tmdb_id)
              return {
                id: movie.id,
                tmdbId: movie.tmdb_id,
                title: movie.title,
                score: movie.score,
                posterPath: tmdbData.poster_path,
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

        setSimilarMovies(enhancedSimilarMovies)
      } catch (err) {
        console.error("Failed to fetch movie data:", err)
        setError("Failed to load movie details. Please try again later.")
      } finally {
        setLoading(false)
        setSimilarMoviesLoading(false)
      }
    }

    fetchMovieData()
  }, [id])

  const handleRateMovie = async (rating: number) => {
    if (!movie) return

    try {
      setRatingLoading(true)
      await rateMovie(movie.id, rating)
      setUserRating(rating)
    } catch (err) {
      console.error("Failed to rate movie:", err)
      alert("Failed to save rating. Please try again.")
    } finally {
      setRatingLoading(false)
    }
  }

  const handleDeleteRating = async () => {
    if (!movie) return

    try {
      setRatingLoading(true)
      await deleteRating(movie.id)
      setUserRating(null)
    } catch (err) {
      console.error("Failed to delete rating:", err)
      alert("Failed to delete rating. Please try again.")
    } finally {
      setRatingLoading(false)
    }
  }

  if (loading) {
    return <LoadingSpinner />
  }

  if (error || !movie) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-white text-center">
          <p className="text-xl">{error || "Movie not found"}</p>
          <button
            onClick={() => window.history.back()}
            className="mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
          >
            Go Back
          </button>
        </div>
      </div>
    )
  }

  const backdropUrl = movie.backdrop_path ? `https://image.tmdb.org/t/p/original${movie.backdrop_path}` : null

  const posterUrl = movie.poster_path
    ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
    : "/placeholder_poster.jpg"

  const director = credits?.crew.find((person) => person.job === "Director")
  const releaseYear = movie.release_date ? new Date(movie.release_date).getFullYear() : "Unknown"
  const runtime = movie.runtime ? `${Math.floor(movie.runtime / 60)}h ${movie.runtime % 60}m` : "Unknown"

  return (
    <div className="relative">
      {/* Backdrop Image */}
      {backdropUrl && (
        <div className="absolute top-0 left-0 w-full h-[500px] z-0">
          <div
            className="w-full h-full bg-cover bg-center"
            style={{
              backgroundImage: `url(${backdropUrl})`,
              boxShadow: "inset 0 -100px 100px -100px #000",
            }}
          >
            <div className="w-full h-full bg-black bg-opacity-70"></div>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row gap-8 mt-8">
          {/* Poster */}
          <div className="w-full md:w-1/3 lg:w-1/4 flex-shrink-0">
            <img
              src={posterUrl || "/placeholder.svg"}
              alt={movie.title}
              className="w-full rounded-lg shadow-lg"
              onError={(e) => {
                e.currentTarget.src = "/placeholder_poster.jpg"
              }}
            />
          </div>

          {/* Details */}
          <div className="flex-1">
            <h1 className="text-4xl font-bold text-white mb-2">
              {movie.title} <span className="text-gray-400">({releaseYear})</span>
            </h1>

            <div className="flex flex-wrap gap-2 mb-4">
              {movie.genres.map((genre) => (
                <span key={genre.id} className="px-3 py-1 bg-gray-800 rounded-full text-sm text-gray-300">
                  {genre.name}
                </span>
              ))}
            </div>

            <div className="flex items-center gap-4 mb-6 text-gray-300">
              <div className="flex items-center">
                <span className="text-yellow-400 mr-1">★</span>
                <span>{movie.vote_average.toFixed(1)}/10</span>
              </div>
              <span>•</span>
              <span>{runtime}</span>
              <span>•</span>
              <span>{movie.release_date}</span>
            </div>

            <div className="mb-6">
              <h2 className="text-xl font-semibold text-white mb-2">Overview</h2>
              <p className="text-gray-300">{movie.overview}</p>
            </div>

            {director && (
              <div className="mb-6">
                <h2 className="text-xl font-semibold text-white mb-2">Director</h2>
                <p className="text-gray-300">{director.name}</p>
              </div>
            )}

            <div className="mb-8">
              <h2 className="text-xl font-semibold text-white mb-4">Your Rating</h2>
              <div className="flex items-center gap-4">
                <StarRating initialRating={userRating || 0} onChange={handleRateMovie} size="lg" />
                {userRating && (
                  <button
                    onClick={handleDeleteRating}
                    disabled={ratingLoading}
                    className="text-red-500 hover:text-red-400 text-sm"
                  >
                    Remove rating
                  </button>
                )}
                {ratingLoading && <span className="text-gray-400 text-sm">Saving...</span>}
              </div>
            </div>
          </div>
        </div>

        {/* Cast */}
        {credits && credits.cast.length > 0 && (
          <div className="mt-12">
            <h2 className="text-2xl font-bold text-white mb-6">Cast</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {credits.cast.slice(0, 6).map((person) => (
                <div key={person.id} className="bg-gray-900 rounded-lg overflow-hidden">
                  <img
                    src={
                      person.profile_path
                        ? `https://image.tmdb.org/t/p/w185${person.profile_path}`
                        : "/placeholder_portrait.png"
                    }
                    alt={person.name}
                    className="w-full aspect-[2/3] object-cover"
                    onError={(e) => {
                      e.currentTarget.src = "/placeholder_portrait.png"
                    }}
                  />
                  <div className="p-3">
                    <h3 className="font-medium text-white">{person.name}</h3>
                    <p className="text-sm text-gray-400">{person.character}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Similar Movies */}
        {!similarMoviesLoading && similarMovies.length > 0 && (
          <div className="mt-12">
            <h2 className="text-2xl font-bold text-white mb-6">Similar Movies You Might Like</h2>
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
        )}
      </div>
    </div>
  )
}

export default MovieDetails
