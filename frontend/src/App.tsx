import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom"
import { AuthProvider } from "./contexts/AuthContext"
import { Suspense, lazy } from "react"
import ProtectedRoute from "./components/ProtectedRoute"
import Navbar from "./components/Navbar"
import LoadingSpinner from "./components/LoadingSpinner"

// Lazy load pages for better performance
const Home = lazy(() => import("./pages/Home"))
const Login = lazy(() => import("./pages/Login"))
const Register = lazy(() => import("./pages/Register"))
const MovieDetails = lazy(() => import("./pages/MovieDetails"))
const SimilarMovies = lazy(() => import("./pages/SimilarMovies"))
const Search = lazy(() => import("./pages/Search"))
const RatedMovies = lazy(() => import("./pages/RatedMovies"))
const CombinedRecommendations = lazy(() => import("./pages/CombinedRecommendations"))

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="app bg-black min-h-screen text-white">
          <Navbar />
          <Suspense fallback={<LoadingSpinner />}>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <Home />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/movie/:id"
                element={
                  <ProtectedRoute>
                    <MovieDetails />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/similar-movies"
                element={<Navigate to="/combined-recommendations" replace />}
              />
              <Route
                path="/search"
                element={
                  <ProtectedRoute>
                    <Search />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/rated-movies"
                element={
                  <ProtectedRoute>
                    <RatedMovies />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/combined-recommendations"
                element={
                  <ProtectedRoute>
                    <CombinedRecommendations />
                  </ProtectedRoute>
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </div>
      </Router>
    </AuthProvider>
  )
}

export default App
