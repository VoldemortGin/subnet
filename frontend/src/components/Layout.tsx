import { NavLink, Outlet } from 'react-router-dom'
import {
  LayoutDashboard,
  ScanSearch,
  SendHorizontal,
  Users,
  ListTodo,
  ShieldAlert,
} from 'lucide-react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/analyze', icon: ScanSearch, label: 'Analyze' },
  { to: '/submit', icon: SendHorizontal, label: 'Submit' },
  { to: '/miners', icon: Users, label: 'Miners' },
  { to: '/tasks', icon: ListTodo, label: 'Tasks' },
]

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <aside className="w-56 shrink-0 bg-[#111] border-r border-white/10 flex flex-col">
        <div className="flex items-center gap-2 px-5 py-6">
          <ShieldAlert className="w-7 h-7 text-red-600" />
          <span className="text-xl font-bold tracking-tight">
            <span className="text-red-600">HARM</span>
          </span>
        </div>

        <nav className="flex-1 px-3 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-red-600/15 text-red-500 border border-red-600/30'
                    : 'text-[#a1a1a1] hover:text-white hover:bg-white/5'
                }`
              }
            >
              <Icon className="w-5 h-5" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="px-5 py-4 border-t border-white/10">
          <p className="text-xs text-[#a1a1a1]">
            Decentralized Forgery Detection
          </p>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto p-8">
        <Outlet />
      </main>
    </div>
  )
}
