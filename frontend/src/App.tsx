import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Analyze from './pages/Analyze'
import Miners from './pages/Miners'
import Tasks from './pages/Tasks'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/miners" element={<Miners />} />
          <Route path="/tasks" element={<Tasks />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
