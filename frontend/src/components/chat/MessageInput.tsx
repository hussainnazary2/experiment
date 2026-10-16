import { useState, useRef, type KeyboardEvent } from 'react'
import { Send, Square, Plus, ChevronDown, FolderOpen, HardDrive } from 'lucide-react'
import { useChatStore } from '../../stores/chatStore'
import { useWorkspaceStore } from '../../stores/workspaceStore'
import { FileMentionPopup } from './FileMentionPopup'
import { ChatAutocomplete, detectTrigger } from './ChatAutocomplete'
import { ModelPicker } from '../model/ModelPicker'

export function MessageInput() {
  const [input, setInput] = useState('')
  const [showMentions, setShowMentions] = useState(false)
  const [mentionQuery, setMentionQuery] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { isStreaming, sendMessage, stopStreaming } = useChatStore()

  const [showAutocomplete, setShowAutocomplete] = useState(false)
  const [acTrigger, setAcTrigger] = useState('')
  const [acQuery, setAcQuery] = useState('')

  // Workspace state (shared with LeftPanel)
  const workspace = useWorkspaceStore((s) => s.workspace)
  const setWorkspace = useWorkspaceStore((s) => s.setWorkspace)
  const [showWorkspacePicker, setShowWorkspacePicker] = useState(false)

  function handleWorkspaceChange(newWorkspace: string) {
    if (newWorkspace === workspace) {
      setShowWorkspacePicker(false)
      return
    }
    setWorkspace(newWorkspace)
    // Clear chat for fresh start
    useChatStore.getState().clearMessages()
    setShowWorkspacePicker(false)
  }

  async function handleBrowseWorkspace() {
    if ((window as any).electronAPI?.openFolderDialog) {
      const path = await (window as any).electronAPI.openFolderDialog()
      if (path) handleWorkspaceChange(path)
    } else {
      const path = prompt('Enter workspace folder path:')
      if (path) handleWorkspaceChange(path)
    }
    setShowWorkspacePicker(false)
  }

  function handleChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    const val = e.target.value
    setInput(val)

    const cursorPos = e.target.selectionStart
    const textBeforeCursor = val.slice(0, cursorPos)
    const atMatch = textBeforeCursor.match(/@([\w.\-/]*)$/)
    if (atMatch) {
      setShowMentions(true)
      setMentionQuery(atMatch[1])
      setShowAutocomplete(false)
      return
    } else {
      setShowMentions(false)
    }

    const trigger = detectTrigger(textBeforeCursor)
    if (trigger) {
      setShowAutocomplete(true)
      setAcTrigger(trigger.trigger)
      setAcQuery(trigger.query)
    } else {
      setShowAutocomplete(false)
    }
  }

  function handleMentionSelect(filePath: string) {
    const cursorPos = textareaRef.current?.selectionStart ?? input.length
    const textBeforeCursor = input.slice(0, cursorPos)
    const textAfterCursor = input.slice(cursorPos)
    const atIndex = textBeforeCursor.lastIndexOf('@')
    const newText = textBeforeCursor.slice(0, atIndex) + `@${filePath} ` + textAfterCursor
    setInput(newText)
    setShowMentions(false)
    textareaRef.current?.focus()
  }

  function handleMentionClose() {
    setShowMentions(false)
    textareaRef.current?.focus()
  }

  function handleAutocompleteSelect(item: any) {
    const cursorPos = textareaRef.current?.selectionStart ?? input.length
    const textBeforeCursor = input.slice(0, cursorPos)
    const textAfterCursor = input.slice(cursorPos)
    const triggerIdx = textBeforeCursor.search(/\/\w*$/)
    const newText = textBeforeCursor.slice(0, triggerIdx) + item.insert + ' ' + textAfterCursor
    setInput(newText)
    setShowAutocomplete(false)
    textareaRef.current?.focus()
  }

  function handleAutocompleteClose() {
    setShowAutocomplete(false)
    textareaRef.current?.focus()
  }

  async function handleSend() {
    const text = input.trim()
    if (!text || isStreaming) return
    setInput('')
    setShowMentions(false)
    await sendMessage(text)
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (showMentions || showAutocomplete) return
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Get short workspace name for display
  const workspaceShort = workspace.split(/[/\\]/).pop() || workspace

  return (
    <div className="px-4 pb-4 pt-2 relative">
      {showMentions && (
        <FileMentionPopup
          query={mentionQuery}
          onSelect={handleMentionSelect}
          onClose={handleMentionClose}
        />
      )}
      {showAutocomplete && (
        <ChatAutocomplete
          query={acQuery}
          trigger={acTrigger}
          onSelect={handleAutocompleteSelect}
          onClose={handleAutocompleteClose}
        />
      )}

      {/* Input container */}
      <div className="bg-elevated border border-border rounded-2xl px-4 py-3 flex items-end gap-2">
        {/* Plus/attach button */}
        <button
          className="p-1 text-text-muted hover:text-text transition-colors mb-0.5"
          title="Attach file"
        >
          <Plus size={18} />
        </button>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Reply..."
          rows={1}
          data-chat-input
          aria-label="Chat message input"
          className="flex-1 bg-transparent text-text placeholder-text-muted outline-none resize-none text-sm"
          style={{ minHeight: '24px', maxHeight: '120px' }}
        />

        {/* Workspace selector */}
        <div className="relative mb-0.5">
          <button
            onClick={() => setShowWorkspacePicker(!showWorkspacePicker)}
            className="flex items-center gap-1 px-2 py-1 text-[11px] text-text-muted hover:text-text-sec rounded-lg hover:bg-bg/50 transition-colors"
            title={`Workspace: ${workspace}`}
          >
            <FolderOpen size={10} />
            <span>{workspaceShort}</span>
            <ChevronDown size={10} />
          </button>

          {showWorkspacePicker && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowWorkspacePicker(false)} />
              <div className="absolute bottom-full right-0 mb-1 w-64 bg-elevated border border-border rounded-xl shadow-xl z-50 overflow-hidden">
                <div className="px-3 py-2 border-b border-border bg-bg/50">
                  <div className="text-[10px] text-text-muted font-medium">Workspace</div>
                  <div className="text-[10px] text-text truncate mt-0.5" title={workspace}>{workspace}</div>
                </div>

                <button
                  onClick={handleBrowseWorkspace}
                  className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-bg/50 transition-colors text-xs text-text"
                >
                  <FolderOpen size={12} className="text-accent" />
                  <span>Browse for workspace...</span>
                </button>

                <button
                  onClick={() => handleWorkspaceChange('.')}
                  className={`w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-bg/50 transition-colors text-xs ${
                    workspace === '.' ? 'text-accent' : 'text-text'
                  }`}
                >
                  <HardDrive size={12} />
                  <span>App directory (default)</span>
                </button>
              </div>
            </>
          )}
        </div>

        {/* Model selector - the shared picker, identical to the chatbot's */}
        <ModelPicker align="top" className="mb-0.5" />

        {/* Send/Stop button */}
        <button
          onClick={isStreaming ? stopStreaming : handleSend}
          disabled={!input.trim() && !isStreaming}
          className={`p-2 rounded-xl transition-colors ${
            isStreaming
              ? 'bg-danger text-white hover:bg-danger/80'
              : input.trim()
                ? 'bg-accent text-onAccent hover:bg-accent-hover'
                : 'bg-border text-text-muted'
          }`}
        >
          {isStreaming ? <Square size={16} /> : <Send size={16} />}
        </button>
      </div>
    </div>
  )
}
