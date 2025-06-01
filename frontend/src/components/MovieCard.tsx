"use client"

import { Link } from "react-router-dom"
import { useState } from "react"

interface MovieCardProps {
  id: number
  tmdbId: number
  title: string
  posterPath?: string
  score?: number
  year?: string
  overview?: string
}

const MovieCard = ({ id, tmdbId, title, posterPath, score, year, overview }: MovieCardProps) => {
  const [isHovered, setIsHovered] = useState(false)
  const [imgError, setImgError] = useState(false)

  // Only try to load from TMDB if we have a valid path
  const posterUrl = posterPath && !imgError 
    ? `https://image.tmdb.org/t/p/w500${posterPath}` 
    : "/placeholder_poster.jpg"

  return (
    <div
      className="relative transition-transform duration-300 ease-in-out"
      style={{
        width: "200px",
        height: "300px",
        transform: isHovered ? "scale(1.05)" : "scale(1)",
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Link to={`/movie/${id}`}>
        <img
          src={posterUrl}
          alt={title}
          className="w-full h-full object-cover rounded-md"
          onError={() => setImgError(true)}
        />

        {isHovered && (
          <div className="absolute inset-0 bg-black bg-opacity-75 p-4 flex flex-col justify-between rounded-md">
            <div>
              <h3 className="text-white font-bold">{title}</h3>
              {year && <p className="text-gray-300 text-sm mt-1">{year}</p>}
              {score !== undefined && (
                <div className="mt-2 flex items-center">
                  <div className="h-2 w-full bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full bg-red-600 rounded-full" style={{ width: `${score * 100}%` }}></div>
                  </div>
                  <span className="ml-2 text-white text-sm">{Math.round(score * 100)}%</span>
                </div>
              )}
              {overview && <p className="text-gray-300 text-xs mt-2 line-clamp-3">{overview}</p>}
            </div>
            <div className="text-white text-sm font-semibold">Click for details</div>
          </div>
        )}
      </Link>
    </div>
  )
}

export default MovieCard
