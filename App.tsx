import React from 'react'
import { StatusBar } from 'react-native'
import { LightScreen } from './src/LightScreen'

function App() {
  return (
    <>
      <StatusBar barStyle="light-content" />
      <LightScreen />
    </>
  )
}

export default App
