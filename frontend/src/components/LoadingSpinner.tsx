interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg"
}

const LoadingSpinner = ({ size = "lg" }: LoadingSpinnerProps) => {
  const sizeClasses = {
    sm: "h-6 w-6",
    md: "h-16 w-16",
    lg: "h-32 w-32",
  }

  const containerClasses = size === "lg" ? "h-screen" : "h-auto"

  return (
    <div className={`flex justify-center items-center ${containerClasses}`}>
      <div className={`animate-spin rounded-full ${sizeClasses[size]} border-t-2 border-b-2 border-red-600`}></div>
    </div>
  )
}

export default LoadingSpinner
