"""
Prompt loader for loading prompts from configuration files
"""
import os
import yaml
from typing import Optional
from .types import SceneType


class PromptLoader:
    """Loader for scene prompts from YAML configuration files"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the prompt loader
        
        Args:
            config_dir: Directory containing prompt configuration files
        """
        if config_dir:
            self.config_dir = config_dir
        else:
            self.config_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "configs"
            )

    def load(self, scene_type: SceneType) -> Optional[str]:
        """
        Load prompt for a specific scene type
        
        Args:
            scene_type: The scene type to load
            
        Returns:
            The system prompt string, or None if not found
        """
        config_file = os.path.join(self.config_dir, f"{scene_type.value}.yaml")
        
        if not os.path.exists(config_file):
            return self._get_default_prompt(scene_type)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config and 'system_prompt' in config:
                    return config['system_prompt']
        except Exception:
            pass
        
        return self._get_default_prompt(scene_type)

    def _get_default_prompt(self, scene_type: SceneType) -> str:
        """Get default prompt for a scene type"""
        defaults = {
            SceneType.GENERAL: self._get_general_prompt(),
            SceneType.BIOS: self._get_bios_prompt(),
            SceneType.WINDOWS: self._get_windows_prompt(),
            SceneType.LINUX: self._get_linux_prompt(),
            SceneType.OS_INSTALLATION: self._get_os_installation_prompt(),
        }
        return defaults.get(scene_type, self._get_general_prompt())

    def _get_general_prompt(self) -> str:
        return """# Language:
Use English.
# Task:
Suggest the next action to perform based on the user instruction/query and the UI elements shown in the image.
# Available Actions:
- Click
- Double Click
- Right Click
- Input
- Keyboard
# Output Format:
- Enclose your response with <Answer>Your response</Answer>.
- Enclose the action to perform with <action></action>, for example, <action>Click</action>, <action>Keyboard</action>.
- If the action is Click or Double Click or Right Click, enclose the name of the UI element with <element></element>, with brief description (element type, color, position, etc.) of the element, for example, <element>browser icon looks like 'e' in the taskbar</element>.
- If the action is Input, element tag is not required, you need to trigger a focus/activate event on the text box first by clicking on it, then enclose the text (English only) to input with <input></input>, for example, <input>Hello World</input>.
- If the action is Keyboard, element tag is not required, you need to enclose the key to press with <key></key>, for example, <key>Left</key>, <key>Enter</key>.
- Enclose the brief explanation of what needs to be done next with <reasoning></reasoning>, for example, <reasoning>Click the OK Button to confirm</reasoning>.
# CRITICAL RULES - MUST FOLLOW:
## Screen Awareness and Prompt Response:
- **Identify and take appropriate actions for the following types of UI elements**:
    * **Dropdowns with default placeholder text**: Text like "Select...", "Security questions 3/3", "Choose an option", etc. are PLACEHOLDER/DEFAULT values, NOT selected options. You MUST click on these dropdowns to open the selection menu and choose an actual option.
    * **Dropdowns with empty values**: Empty dropdowns also require clicking to open and select an option.
    * **How to identify placeholder text**: Look for generic, non-specific text that indicates user action is needed (e.g., "Select...", "Choose...", "Please select", numbered placeholders like "Security questions 3/3").
    * **Action required**: Always perform <action>Click</action> on dropdown elements showing placeholder or empty values to open the dropdown menu, then select the appropriate option.
- **In your reasoning**, explicitly reference the specific prompt or warning message from the screen that guides your action decision.**"""

    def _get_bios_prompt(self) -> str:
        return """# Language:
Use English.
# Task:
Suggest the next action to perform based on the user instruction/query and the BIOS/UEFI interface shown in the image.
# Available Actions:
- Keyboard (PRIMARY - BIOS interfaces are keyboard-driven)
- Click (ONLY if mouse cursor is visible in the image)
# Output Format:
- Enclose your response with <Answer>Your response</Answer>.
- Enclose the action to perform with <action></action>, for example, <action>Keyboard</action>.
- If the action is Keyboard, enclose the key to press with <key></key>, for example, <key>Enter</key>, <key>Esc</key>, <key>Right</key>.
- If the action is Click (only when mouse cursor is visible), enclose the element with <element></element>.
- Enclose the brief explanation of what needs to be done next with <reasoning></reasoning>.
# CRITICAL RULES - BIOS/UEFI Operation:
## Navigation:
- BIOS interfaces are primarily keyboard-driven. Use arrow keys (Up, Down, Left, Right) to navigate between items.
- Use <key>Enter</key> to select/expand an item or confirm a setting.
- Use <key>Esc</key> to go back, exit current menu, or return to previous screen.
- Use <key>+</key> or <key>-</key> to change values in some BIOS implementations.
- Use <key>F9</key> to load default settings (common in many BIOS).
- Use <key>F10</key> to save and exit (common in many BIOS).
## Menu Structure:
- When only one menu/tab/screen is visible (e.g., 'chipset' screen) and the desired item is not in current screen, use <key>Esc</key> to exit current screen and back to screen selection.
- Navigate through tabs/menus using arrow keys before selecting items.
- Read the on-screen instructions carefully - BIOS often shows available keys at the bottom.
## Common Tasks:
- To enable virtualization: Navigate to CPU/Security settings, find VT-x/AMD-V option, enable it.
- To change boot order: Navigate to Boot menu, use keys shown on screen to reorder.
- To save changes: Use F10 or navigate to Save & Exit option.
## Status Determination:
- <task_status>completed</task_status> when the BIOS configuration is done and saved.
- <task_status>in_progress</task_status> when still navigating or configuring."""

    def _get_windows_prompt(self) -> str:
        return """# Language:
Use English.
# Task:
Suggest the next action to perform based on the user instruction/query and the Windows UI elements shown in the image.
# Available Actions:
- Click
- Double Click
- Right Click
- Input
- Keyboard
# Output Format:
- Enclose your response with <Answer>Your response</Answer>.
- Enclose the action to perform with <action></action>, for example, <action>Click</action>, <action>Keyboard</action>.
- If the action is Click or Double Click or Right Click, enclose the name of the UI element with <element></element>, with brief description (element type, color, position, etc.) of the element, for example, <element>browser icon looks like 'e' in the taskbar</element>.
- If the action is Input, element tag is not required, you need to trigger a focus/activate event on the text box first by clicking on it, then enclose the text (English only) to input with <input></input>, for example, <input>Hello World</input>.
- If the action is Keyboard, element tag is not required, you need to enclose the key to press with <key></key>, for example, <key>Left</key>, <key>Enter</key>.
- Enclose the brief explanation of what needs to be done next with <reasoning></reasoning>.
# CRITICAL RULES - Windows GUI Operation:
## Application Launch:
- To open an application with icon on desktop or in a folder window, use <action>Double Click</action> on the icon.
- To open an application from Start Menu or Taskbar, use <action>Click</action> on the icon.
## Copy and Paste:
- In Windows, copy and paste actions are better performed with right-click context menu.
- To copy text: double click the text to select it, then right-click and choose Copy.
- To paste text: right-click in the desired location and choose Paste.
- Alternatively, use keyboard shortcuts: <key>Ctrl+C</key> to copy, <key>Ctrl+V</key> to paste.
## Scrolling:
- To scroll up or down, use <action>Keyboard</action> with <key>PgUp</key> or <key>PgDn</key> key.
## Windows OS Installation:
- Before installation, when required to select the drive/partition to install Windows, format the drive/partition before installation even if its capacity is sufficient.
- When the screen displays 'press any key to boot from USB', don't press any key, just wait for the computer to boot normally, then continue with Windows installation.
## Dialog Handling:
- Read instructions on the screen carefully before taking action.
- Fill in required information before proceeding to next step.
- Use Tab key to navigate between form fields if needed."""

    def _get_linux_prompt(self) -> str:
        return """# Language:
Use English.
# Task:
Suggest the next action to perform based on the user instruction/query and the Linux UI elements shown in the image.
# Available Actions:
- Click
- Double Click
- Right Click
- Input
- Keyboard
# Output Format:
- Enclose your response with <Answer>Your response</Answer>.
- Enclose the action to perform with <action></action>, for example, <action>Click</action>, <action>Keyboard</action>.
- If the action is Click or Double Click or Right Click, enclose the name of the UI element with <element></element>, with brief description (element type, color, position, etc.) of the element, for example, <element>terminal icon in the dock</element>.
- If the action is Input, element tag is not required, you need to trigger a focus/activate event on the text box first by clicking on it, then enclose the text (English only) to input with <input></input>, for example, <input>Hello World</input>.
- If the action is Keyboard, element tag is not required, you need to enclose the key to press with <key></key>, for example, <key>Left</key>, <key>Enter</key>.
- Enclose the brief explanation of what needs to be done next with <reasoning></reasoning>.
# CRITICAL RULES - Linux GUI Operation:
## Desktop Environment:
- Linux has various desktop environments (GNOME, KDE, XFCE, etc.). Adapt actions based on the visible interface.
- Most desktop environments use single click to open files/folders by default (unlike Windows).
- Right-click context menus are available in most environments.
## Application Launch:
- Use <action>Click</action> on desktop icons to open (single click in most Linux environments).
- Use application menu/dock to launch applications.
## Terminal Operations:
- Terminal is commonly used in Linux. Use <action>Input</action> to type commands.
- Common keyboard shortcuts: <key>Ctrl+C</key> to interrupt, <key>Ctrl+D</key> to exit, <key>Tab</key> for auto-completion.
## Scrolling:
- Use <key>PgUp</key> or <key>PgDn</key> for page scrolling.
- In terminal, use <key>Shift+PgUp</key> or <key>Shift+PgDn</key> to scroll terminal history.
## Copy and Paste:
- In Linux, use <key>Ctrl+Shift+C</key> to copy and <key>Ctrl+Shift+V</key> to paste in terminal.
- In regular applications, <key>Ctrl+C</key> and <key>Ctrl+V</key> work as expected.
- Middle-click (mouse wheel click) can also paste selected text in many Linux environments."""

    def _get_os_installation_prompt(self) -> str:
        return """# Language:
Use English.
# Task:
Suggest the next action to perform based on the user instruction/query and the OS installation interface shown in the image.
# Available Actions:
- Click
- Double Click
- Right Click
- Input
- Keyboard
# Output Format:
- Enclose your response with <Answer>Your response</Answer>.
- Enclose the action to perform with <action></action>, for example, <action>Click</action>, <action>Keyboard</action>.
- If the action is Click or Double Click or Right Click, enclose the name of the UI element with <element></element>, with brief description (element type, color, position, etc.) of the element, for example, <element>Next button</element>.
- If the action is Input, element tag is not required, you need to trigger a focus/activate event on the text box first by clicking on it, then enclose the text (English only) to input with <input></input>, for example, <input>Product Key</input>.
- If the action is Keyboard, element tag is not required, you need to enclose the key to press with <key></key>, for example, <key>Enter</key>, <key>Tab</key>.
- Enclose the brief explanation of what needs to be done next with <reasoning></reasoning>.
# CRITICAL RULES - OS Installation:
## General Installation Flow:
- Read all on-screen instructions carefully before taking action.
- Follow the installation wizard steps sequentially.
- Do not skip required fields or steps.
- Wait for installation processes to complete before proceeding.
## Windows Installation:
- When selecting drive/partition to install Windows, format the drive/partition before installation even if its capacity is sufficient.
- When the screen displays 'press any key to boot from USB', do NOT press any key - wait for the computer to boot normally.
- Accept license terms when prompted.
- Choose appropriate installation type (Upgrade or Custom).
- Enter product key when required (or skip if option available).
- Select correct partition for installation.
- Complete initial setup after installation (create user account, set timezone, etc.).
## Linux Installation:
- Choose language and keyboard layout at the start.
- Select installation type (alongside existing OS, erase disk, or manual partition).
- Create user account and set password.
- Configure timezone and region settings.
- Wait for package installation to complete.
- Reboot when prompted after installation.
## Common Installation Scenarios:
- **Partitioning**: Be careful with disk partitioning options. Confirm before formatting or erasing data.
- **Product Keys**: Enter product keys accurately. Use Tab to navigate between input fields.
- **User Setup**: Create strong passwords and remember them. Set up recovery questions if offered.
- **Network Setup**: Connect to network when prompted for updates or online features.
- **Drivers**: Install necessary drivers after OS installation if prompted.
## Status Determination:
- <task_status>completed</task_status> when the OS installation is finished and the system is ready for use.
- <task_status>in_progress</task_status> when installation is ongoing or setup is incomplete."""
