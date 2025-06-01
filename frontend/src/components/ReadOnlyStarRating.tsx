"use client"

interface ReadOnlyStarRatingProps {
  rating: number
  size?: "sm" | "md" | "lg"
}

const ReadOnlyStarRating = ({ rating, size = "md" }: ReadOnlyStarRatingProps) => {
  const sizeClasses = {
    sm: "text-xl",
    md: "text-2xl",
    lg: "text-3xl",
  }

  return (
    <div className="flex">
      {[1, 2, 3, 4, 5].map((star) => (
        <span
          key={star}
          className={`${sizeClasses[size]} ${rating >= star ? "text-yellow-400" : "text-gray-500"}`}
        >
          ★
        </span>
      ))}
    </div>
  )
}

export default ReadOnlyStarRating 