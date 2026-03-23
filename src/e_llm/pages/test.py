"""Test tab — streaming chat against the running llama-server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from e_llm.core.state import State


def create(s: State) -> None:
    """Build test chat with server guard and model info."""
    history: list[dict[str, str]] = []

    # Model info bar
    model_bar = ui.row().classes("w-full items-center gap-2 mb-2")

    with ui.expansion("System Prompt", icon="psychology").classes("w-full"):
        system_prompt = (
            ui.textarea(
                value="You are a helpful assistant.",
            )
            .props("outlined autogrow")
            .classes("w-full")
        )

    chat_area = ui.scroll_area().classes("w-full").style("height: 52vh")

    disabled_banner = ui.card().classes("w-full").style("border-color: orange !important")
    disabled_banner.visible = False
    with disabled_banner, ui.row().classes("items-center gap-2"):
        ui.icon("warning", color="orange").classes("text-lg")
        ui.label("Server is not running — go to Configuration to load a model and start it.").classes("text-caption")

    with ui.row().classes("w-full items-end gap-2 no-wrap mt-2"):
        msg_input = ui.input(placeholder="Type a message...").props("outlined rounded dense").classes("flex-grow")
        clear_btn = ui.button(icon="delete_sweep").props("flat round dense").style("color: var(--accent)")
        send_btn = ui.button(icon="send").props("round dense unelevated").style("background: var(--accent) !important")

    async def _handle_send() -> None:
        if not s.server_manager.is_running:
            disabled_banner.visible = True
            return

        text = msg_input.value.strip()
        if not text:
            return
        msg_input.value = ""
        disabled_banner.visible = False

        history.append({"role": "user", "content": text})

        with chat_area:
            ui.chat_message(text=text, name="You", sent=True)
            response_md = ui.markdown("*Generating...*")

        messages = [{"role": "system", "content": system_prompt.value}] + history

        full_response = ""
        try:
            async for token in s.adapter.stream_completion(messages=messages):
                full_response += token
                response_md.set_content(full_response)
        except Exception as exc:
            full_response = f"**Error:** {exc}"
            response_md.set_content(full_response)

        if not full_response:
            full_response = "*No response received.*"
            response_md.set_content(full_response)

        history.append({"role": "assistant", "content": full_response})
        chat_area.scroll_to(percent=1.0)

    def _handle_clear() -> None:
        history.clear()
        chat_area.clear()
        disabled_banner.visible = False

    def _update_guard() -> None:
        running = s.server_manager.is_running
        msg_input.set_enabled(running)
        send_btn.set_enabled(running)
        disabled_banner.visible = not running

        model_bar.clear()
        with model_bar:
            if running:
                ui.icon("smart_toy").style("color: var(--accent)")
                ui.label("Model loaded").classes("text-caption").style("color: var(--text-dim)")
            else:
                ui.icon("smart_toy", color="grey")
                ui.label("No model loaded").classes("text-caption text-grey")

    send_btn.on_click(_handle_send)
    clear_btn.on_click(_handle_clear)
    msg_input.on("keydown.enter", _handle_send)
    ui.timer(3.0, _update_guard)
    _update_guard()
