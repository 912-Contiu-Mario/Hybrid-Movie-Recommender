"use client"

import { useRef } from "react"
import MovieCard from "./MovieCard"

interface Movie {
  id: number
  tmdbId: number
  title: string
  posterPath?: string
  score?: number
  year?: string
  overview?: string
}

interface MovieRowProps {
  title: string
  movies: Movie[]
}

const MovieRow = ({ title, movies }: MovieRowProps) => {
  const rowRef = useRef<HTMLDivElement>(null)

  const scroll = (direction: "left" | "right") => {
    if (rowRef.current) {
      const { current } = rowRef
      const scrollAmount = direction === "left" ? -current.offsetWidth : current.offsetWidth
      current.scrollBy({ left: scrollAmount, behavior: "smooth" })
    }
  }

  if (!movies.length) {
    return null
  }

  return (
    <div className="mb-8">
      <h2 className="text-xl font-bold mb-4 text-white">{title}</h2>
      <div className="relative group">
        <button
          className="absolute left-0 top-1/2 transform -translate-y-1/2 bg-black bg-opacity-50 text-white p-2 rounded-full z-10 opacity-0 group-hover:opacity-100 transition-opacity"
          onClick={() => scroll("left")}
        >
          ◀
        </button>

        <div
          ref={rowRef}
          className="flex space-x-4 overflow-x-auto pb-4 scrollbar-hide"
          style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
        >
          {movies.map((movie) => (
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

        <button
          className="absolute right-0 top-1/2 transform -translate-y-1/2 bg-black bg-opacity-50 text-white p-2 rounded-full z-10 opacity-0 group-hover:opacity-100 transition-opacity"
          onClick={() => scroll("right")}
        >
          ▶
        </button>
      </div>
    </div>
  )
}

export default MovieRow
