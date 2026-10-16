import { create } from 'zustand'

// The center column shows either the plain testing chatbot ("chat") or the
// agent chat ("agent"). Chatbot is the default: this build is a local-model
// speed demo, not an agent run.
type CenterView = 'chat' | 'agent'

interface UIState {
  theme: 'dark' | 'light'
  leftPanelOpen: boolean
  rightPanelOpen: boolean
  rightPanelTab: string
  agentMode: boolean
  centerView: CenterView
  toggleTheme: () => void
  toggleLeftPanel: () => void
  toggleRightPanel: () => void
  setRightPanelTab: (tab: string) => void
  toggleAgentMode: () => void
  setCenterView: (view: CenterView) => void
}

export const useUIStore = create<UIState>((set) => ({
  theme: 'light',
  leftPanelOpen: true,
  rightPanelOpen: true,
  rightPanelTab: 'files',
  agentMode: true,
  centerView: 'chat',

  toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
  toggleLeftPanel: () => set((s) => ({ leftPanelOpen: !s.leftPanelOpen })),
  toggleRightPanel: () => set((s) => ({ rightPanelOpen: !s.rightPanelOpen })),
  setRightPanelTab: (tab) => set({ rightPanelTab: tab }),
  toggleAgentMode: () => set((s) => ({ agentMode: !s.agentMode })),
  setCenterView: (view) => set({ centerView: view }),
}))
