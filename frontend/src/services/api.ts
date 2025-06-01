import axios from "axios"

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
const TMDB_API_KEY = import.meta.env.VITE_TMDB_API_KEY
const TMDB_BASE_URL = "https://api.themoviedb.org/3"

// Create axios instances
const api = axios.create({
  baseURL: API_BASE_URL,
})

const tmdbApi = axios.create({
  baseURL: TMDB_BASE_URL,
  params: {
    api_key: TMDB_API_KEY,
  },
})

// Export instances for direct use
export { api, tmdbApi }

// Add token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("token")
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error),
)

// Auth endpoints
export const registerUser = async (username: string, email: string, password: string) => {
  const response = await api.post("/auth/register", { username, email, password })
  return response.data
}

export const loginUser = async (username: string, password: string) => {
  const formData = new URLSearchParams()
  formData.append("username", username)
  formData.append("password", password)

  const response = await api.post("/auth/login", formData, {
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
  })
  return response.data
}

export const getCurrentUser = async (token: string) => {
  const response = await api.get("/users/me", {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })
  return response.data
}

// User ratings
export const getUserRatings = async () => {
  const response = await api.get("/users/ratings")
  return response.data
}

export const rateMovie = async (movieId: number, rating: number) => {
  const data = {
    movie_id: movieId,
    rating,
  }
  console.log('Sending rating data:', JSON.stringify(data))
  try {
    const response = await api.post("/users/ratings", data)
    console.log('Rating response:', response.data)
    return response.data
  } catch (error) {
    console.error('Error rating movie:', error)
    throw error
  }
}

export const deleteRating = async (movieId: number) => {
  const response = await api.delete(`/users/ratings/${movieId}`)
  return response.data
}

// Recommendations
export const getLightGCNRecommendations = async (userId: number, numRecs = 10) => {
  const response = await api.get(`/recommendations/lightgcn/${userId}`, {
    params: { num_recs: numRecs },
  })
  return response.data
}

export const getContentBasedRecommendations = async (userId: number, numRecs = 10) => {
  const response = await api.get(`/recommendations/content/${userId}`, {
    params: { num_recs: numRecs },
  })
  return response.data
}

export const getHybridRecommendations = async (userId: number, numRecs = 10) => {
  const response = await api.get(`/recommendations/hybrid/${userId}`, {
    params: { num_recs: numRecs },
  })
  return response.data
}

export const getSimilarMovies = async (movieId: number, numSimilar = 10) => {
  const response = await api.get(`/recommendations/similar-movies/${movieId}`, {
    params: { num_similar: numSimilar },
  })
  return response.data
}

export const getCombinedRecommendations = async (movieIds: number[], numRecs = 10) => {
  const response = await api.post(`/recommendations/generate-recs-for-several-items`, 
    movieIds,
    {
      params: { num_recs: numRecs },
    }
  )
  return response.data
}

// TMDB API
export const getMovieDetails = async (tmdbId: number) => {
  console.log(`Fetching details for TMDB ID: ${tmdbId}`)
  try {
    // First check if we have an internal ID for this TMDB ID
    const internalResponse = await api.get(`/movies/tmdb/${tmdbId}`)
    const movieData = internalResponse.data
    console.log('Internal movie data:', movieData)
    
    // Then get TMDB details
    const tmdbResponse = await tmdbApi.get(`/movie/${tmdbId}`)
    const tmdbData = tmdbResponse.data
    
    // Combine data
    return {
      ...tmdbData,
      id: movieData.id // Make sure to use internal ID
    }
  } catch (error) {
    console.error('Error fetching movie details:', error)
    // If we can't get internal data, just use TMDB data
    const tmdbResponse = await tmdbApi.get(`/movie/${tmdbId}`)
    return tmdbResponse.data
  }
}

export const getMovieByInternalId = async (internalId: number) => {
  try {
    // Get the movie details from our database first
    const internalResponse = await api.get(`/movies/${internalId}`)
    const movieData = internalResponse.data
    
    // Then get TMDB details using the tmdb_id
    const tmdbResponse = await tmdbApi.get(`/movie/${movieData.tmdb_id}`)
    const tmdbData = tmdbResponse.data
    
    // Combine data
    return {
      ...tmdbData,
      id: internalId, // Make sure to use internal ID
      tmdbId: movieData.tmdb_id
    }
  } catch (error) {
    console.error('Error fetching movie by internal ID:', error)
    throw error
  }
}

export const getMovieCredits = async (tmdbId: number) => {
  const response = await tmdbApi.get(`/movie/${tmdbId}/credits`)
  return response.data
}

export const searchMovies = async (query: string) => {
  const response = await tmdbApi.get("/search/movie", {
    params: { query },
  })
  return response.data.results
}

// New function to filter TMDB IDs
export const filterExistingTmdbIds = async (tmdbIds: number[]) => {
  const response = await api.post(
    "/movies/filter-existing-TMDB-ids",
    {
      tmdb_ids: tmdbIds
    }
  )
  return response.data // Now returns [{id: number, tmdb_id: number}, ...]
}

// Enhanced search that only returns movies in your database
export const searchExistingMovies = async (query: string) => {
  // Get results from TMDB
  const tmdbResults = await searchMovies(query)
  
  if (tmdbResults.length === 0) {
    return []
  }
  
  // Extract TMDB IDs
  const tmdbIds = tmdbResults.map((movie: {id: number}) => movie.id)
  
  // Filter to get only existing IDs and their internal IDs
  const existingMoviesMap = await filterExistingTmdbIds(tmdbIds)
  
  // Create a mapping from TMDB ID to internal ID
  const tmdbToInternalMap = new Map(
    existingMoviesMap.map((movie: {id: number, tmdb_id: number}) => [movie.tmdb_id, movie.id])
  )
  
  // Return only movies that exist in your database, with internal IDs
  return tmdbResults
    .filter((movie: {id: number}) => tmdbToInternalMap.has(movie.id))
    .map((movie: any) => ({
      ...movie,
      id: tmdbToInternalMap.get(movie.id), // Use internal ID as id
      tmdbId: movie.id // Keep TMDB ID as tmdbId
    }))
}

export const getFullMovieDetails = async (movieId: number, tmdbId: number) => {
  const [tmdbDetails, credits] = await Promise.all([getMovieDetails(tmdbId), getMovieCredits(tmdbId)])

  return {
    id: movieId,
    tmdbId,
    ...tmdbDetails,
    credits,
  }
}

// Get TMDB details for a movie we already know exists in our database
export const getTMDBDetails = async (tmdbId: number) => {
  try {
    const response = await tmdbApi.get(`/movie/${tmdbId}`)
    return response.data
  } catch (error) {
    console.error('Error fetching TMDB details:', error)
    throw error
  }
}
