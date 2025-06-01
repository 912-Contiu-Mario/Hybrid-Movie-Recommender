"use client"

import { useState } from "react"

interface StarRatingProps {
  initialRating?: number
  onChange: (rating: number) => void
  size?: "sm" | "md" | "lg"
}

const StarRating = ({ initialRating = 0, onChange, size = "md" }: StarRatingProps) => {
  const [rating, setRating] = useState(initialRating)
  const [hover, setHover] = useState(0)

  const sizeClasses = {
    sm: "text-xl",
    md: "text-2xl",
    lg: "text-3xl",
  }

  const handleClick = (value: number) => {
    setRating(value)
    onChange(value)
  }

  return (
    <div className="flex">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          className={`${sizeClasses[size]} focus:outline-none transition-colors duration-200`}
          onClick={() => handleClick(star)}
          onMouseEnter={() => setHover(star)}
          onMouseLeave={() => setHover(0)}
        >
          <span className={`${(hover || rating) >= star ? "text-yellow-400" : "text-gray-500"}`}>★</span>
        </button>
      ))}
    </div>
  )
}

export default StarRating
