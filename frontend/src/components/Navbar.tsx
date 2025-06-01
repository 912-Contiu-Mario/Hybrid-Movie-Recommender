"use client"

import type React from "react"

import { Link, useNavigate } from "react-router-dom"
import { useAuth } from "../contexts/AuthContext"
import { useState } from "react"

const Navbar = () => {
  const { isAuthenticated, logout, user } = useAuth()
  const navigate = useNavigate()
  const [searchQuery, setSearchQuery] = useState("")
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      navigate(`/search?q=${encodeURIComponent(searchQuery)}`)
    }
  }

  const handleLogout = () => {
    logout()
    navigate("/login")
  }

  return (
    <nav className="bg-black py-4 px-6 flex items-center justify-between border-b border-gray-800">
      <div className="flex items-center">
        <Link to="/" className="text-red-600 font-bold text-2xl mr-10">
          MovieRecs
        </Link>
        {isAuthenticated && (
          <div className="hidden md:flex space-x-6">
            <Link to="/" className="text-gray-300 hover:text-white">
              Home
            </Link>
            <Link to="/rated-movies" className="text-gray-300 hover:text-white">
              My Ratings
            </Link>
            <Link to="/combined-recommendations" className="text-gray-300 hover:text-white">
              Magic Recommender
            </Link>
          </div>
        )}
      </div>

      <div className="flex items-center space-x-4">
        {isAuthenticated && (
          <>
            <form onSubmit={handleSearch} className="relative">
              <input
                type="text"
                placeholder="Search movies..."
                className="bg-gray-800 text-white px-4 py-2 rounded-full w-full md:w-64 focus:outline-none focus:ring-2 focus:ring-red-600"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <button type="submit" className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400">
                🔍
              </button>
            </form>
            <div className="relative">
              <button 
                className="flex items-center text-gray-300 hover:text-white"
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                onBlur={() => setTimeout(() => setIsDropdownOpen(false), 100)}
              >
                <span className="mr-2">{user?.username}</span>
                <span>{isDropdownOpen ? '▲' : '▼'}</span>
              </button>
              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-gray-900 rounded shadow-lg py-2 z-10">
                  <button
                    onClick={handleLogout}
                    className="block px-4 py-2 text-gray-300 hover:bg-gray-800 hover:text-white w-full text-left"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          </>
        )}
        {!isAuthenticated && (
          <div className="space-x-4">
            <Link to="/login" className="text-gray-300 hover:text-white">
              Login
            </Link>
            <Link to="/register" className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700">
              Sign Up
            </Link>
          </div>
        )}
      </div>
    </nav>
  )
}

export default Navbar
