import { Header } from './Header'
import { LeftPanel } from './LeftPanel'
import { RightPanel } from './RightPanel'
import { MobileNav } from './MobileNav'
import { ChatPanel } from '../chat/ChatPanel'
import { PlaygroundChat } from '../chat/PlaygroundChat'
import { CommandPalette } from '../ui/CommandPalette'
import { VersionBanner } from '../ui/VersionBanner'
import { CompatibilityDialog } from '../model/CompatibilityDialog'
import { AriaLive } from '../ui/AriaLive'
import { OfflineIndicator } from '../ui/OfflineIndicator'
import { UpdatePanel } from '../ui/UpdatePanel'
import { NotificationPanel } from '../ui/NotificationPanel'
import { HardwareHUD } from '../ui/HardwareHUD'
import { useUIStore } from '../../stores/uiStore'
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts'
import { useAccessibilityShortcuts } from '../../hooks/useAccessibility'
import { useResponsive } from '../../hooks/useResponsive'
import { initTheme } from '../../stores/themeStore'

// Initialize theme on module load
initTheme()

export function AppLayout() {
  const { leftPanelOpen, rightPanelOpen, centerView, setCenterView } = useUIStore()
  const { isMobile } = useResponsive()

  // Register and listen for keyboard shortcuts
  useKeyboardShortcuts()
  useAccessibilityShortcuts()

  return (
    <div className="h-screen flex flex-col bg-bg">
      <AriaLive />
      <Header />
      <VersionBanner />
      <div className="flex-1 flex overflow-hidden" id="main-content" role="main">
        {/* Left panel: hidden on mobile, visible on tablet+ */}
        {leftPanelOpen && !isMobile && <LeftPanel />}
        <div className="flex-1 min-w-0 flex flex-col">
          {/* Center view switch: the plain testing chatbot (default) or the agent. */}
          <div className="flex items-center gap-1 px-3 py-1.5 border-b border-border bg-surface">
            <div className="flex gap-0.5 bg-elevated rounded-lg p-0.5">
              <button
                onClick={() => setCenterView('chat')}
                className={`text-[11px] font-medium px-3 py-1 rounded-md transition-colors ${centerView === 'chat' ? 'bg-accent/15 text-accent' : 'text-text-muted hover:text-text'}`}
              >
                Chatbot
              </button>
              <button
                onClick={() => setCenterView('agent')}
                className={`text-[11px] font-medium px-3 py-1 rounded-md transition-colors ${centerView === 'agent' ? 'bg-accent/15 text-accent' : 'text-text-muted hover:text-text'}`}
              >
                Agent
              </button>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            {centerView === 'chat' ? <PlaygroundChat /> : <ChatPanel />}
          </div>
        </div>
        {/* Right panel: hidden on mobile, overlay on tablet, side on desktop */}
        {rightPanelOpen && !isMobile && <RightPanel />}
      </div>
      {/* Mobile bottom navigation */}
      <MobileNav />
      {/* Live hardware HUD — RAM/CPU/GPU always visible in recordings */}
      {!isMobile && <HardwareHUD />}
      <CommandPalette />
      <CompatibilityDialog />
      <OfflineIndicator />
      <NotificationPanel />
      <UpdatePanel />
    </div>
  )
}
