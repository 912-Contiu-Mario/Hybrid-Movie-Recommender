"use client"

import { useEffect, useState } from "react"
import { useAuth } from "../contexts/AuthContext"
import {
  getLightGCNRecommendations,
  getContentBasedRecommendations,
  getHybridRecommendations,
  getTMDBDetails,
} from "../services/api"
import MovieCard from "../components/MovieCard"
import LoadingSpinner from "../components/LoadingSpinner"

interface Movie {
  id: number
  tmdbId: number
  title: string
  score: number
  posterPath?: string
  releaseDate?: string
  overview?: string
  year?: string
}

// Tab type definition
type RecommendationType = 'hybrid' | 'lightgcn' | 'content'

const Home = () => {
  const { user } = useAuth()
  const [lightGCNRecs, setLightGCNRecs] = useState<Movie[]>([])
  const [contentRecs, setContentRecs] = useState<Movie[]>([])
  const [hybridRecs, setHybridRecs] = useState<Movie[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [activeTab, setActiveTab] = useState<RecommendationType>("hybrid")

  useEffect(() => {
    const fetchRecommendations = async () => {
      if (!user) return

      try {
        setLoading(true)

        // Fetch all recommendation types
        const [lightGCNData, contentData, hybridData] = await Promise.all([
          getLightGCNRecommendations(user.id, 20),
          getContentBasedRecommendations(user.id, 20),
          getHybridRecommendations(user.id, 20),
        ])

        // Process and enhance movie data with TMDB details
        const enhanceLightGCNRecs = await enhanceMoviesWithTMDBData(lightGCNData)
        const enhanceContentRecs = await enhanceMoviesWithTMDBData(contentData)
        const enhanceHybridRecs = await enhanceMoviesWithTMDBData(hybridData)

        setLightGCNRecs(enhanceLightGCNRecs)
        setContentRecs(enhanceContentRecs)
        setHybridRecs(enhanceHybridRecs)
      } catch (err) {
        console.error("Failed to fetch recommendations:", err)
        setError("Failed to load recommendations. Please try again later.")
      } finally {
        setLoading(false)
      }
    }

    fetchRecommendations()
  }, [user])

  // Helper function to enhance movie data with TMDB details
  const enhanceMoviesWithTMDBData = async (movies: any[]) => {
    const enhancedMovies = await Promise.all(
      movies.map(async (movie) => {
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

    return enhancedMovies
  }

  const getActiveRecommendations = () => {
    switch (activeTab) {
      case "hybrid":
        return {
          title: "Recommended for You",
          description: "Smart recommendations combining collaborative and content-based filtering",
          movies: hybridRecs
        }
      case "lightgcn":
        return {
          title: "Based on Your Taste",
          description: "Recommendations based on rating patterns of users with similar tastes",
          movies: lightGCNRecs
        }
      case "content":
        return {
          title: "Similar to Movies You Like",
          description: "Recommendations based on movie content and features you've enjoyed",
          movies: contentRecs
        }
      default:
        return {
          title: "Recommended for You",
          description: "Smart recommendations tailored just for you",
          movies: hybridRecs
        }
    }
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
            onClick={() => window.location.reload()}
            className="mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  const activeRecommendations = getActiveRecommendations()

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">Welcome back, {user?.username}</h1>
        <p className="text-gray-400">Here are your personalized movie recommendations</p>
      </div>

      {/* Tabs Navigation */}
      <div className="mb-8 border-b border-gray-700">
        <div className="flex space-x-1">
          <button
            onClick={() => setActiveTab("hybrid")}
            className={`px-4 py-2 font-medium rounded-t-lg transition-colors ${
              activeTab === "hybrid"
                ? "bg-red-600 text-white"
                : "text-gray-400 hover:text-white hover:bg-gray-800"
            }`}
          >
            Smart Recommendations
          </button>
          <button
            onClick={() => setActiveTab("lightgcn")}
            className={`px-4 py-2 font-medium rounded-t-lg transition-colors ${
              activeTab === "lightgcn"
                ? "bg-red-600 text-white"
                : "text-gray-400 hover:text-white hover:bg-gray-800"
            }`}
          >
            Based on Your Taste
          </button>
          <button
            onClick={() => setActiveTab("content")}
            className={`px-4 py-2 font-medium rounded-t-lg transition-colors ${
              activeTab === "content"
                ? "bg-red-600 text-white"
                : "text-gray-400 hover:text-white hover:bg-gray-800"
            }`}
          >
            Similar to Your Favorites
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">{activeRecommendations.title}</h2>
        <p className="text-gray-400 mb-6">{activeRecommendations.description}</p>
      </div>

      {/* Grid Layout for Movies */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
        {activeRecommendations.movies.map((movie) => (
          <div key={movie.id} className="flex flex-col">
            <MovieCard
              id={movie.id}
              tmdbId={movie.tmdbId}
              title={movie.title}
              posterPath={movie.posterPath}
              score={movie.score}
              year={movie.year}
              overview={movie.overview}
            />
            <div className="mt-2">
              <div className="flex items-center justify-between">
                <p className="text-white font-medium truncate">{movie.title}</p>
                <span className="text-red-600 font-bold">{Math.round(movie.score * 100)}%</span>
              </div>
              <p className="text-gray-400 text-sm">{movie.year}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default Home
