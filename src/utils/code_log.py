"""
 @file code_log.py

 @brief Utility module providing enhanced console logging with caller context.

 @details Captures filename, function, and line information for log messages while
 routing errors and warnings separately from standard informational output.

 *Version History*:
 - Created: 2018-10-07

 @author Thomas Gumbricht

 @date Created: 2018-10-07
"""
import inspect
import sys
from datetime import datetime

def Log(message, space='          ', level='INFO'):
    """
    @brief Logs formatted messages to console with automatic file location and caller context information.

    @details
    Extracts the calling context using Python's inspect module and formats log messages with:
    - Timestamp (HH:MM:SS)
    - Severity level (DEBUG / INFO / WARNING / ERROR)
    - Source filename (basename only)
    - Calling function or method name ('main' for module-level calls)
    - Line number where Log() was called

    ERROR and WARNING messages are written to stderr; DEBUG and INFO go to stdout.
    The clickable file URL is only appended for WARNING and ERROR levels to reduce noise.

    @param message  String containing the log message to display. Can be multi-line.
    @param space    String used as left indentation/padding for all log output lines.
                    Default is 10 spaces ('          '). Used for visual alignment in console output.
    @param level    Severity level string: 'DEBUG', 'INFO', 'WARNING', or 'ERROR'.
                    Default is 'INFO'.
    @return None. Outputs formatted message directly to console via print().
    """

    frame  = inspect.currentframe().f_back
    code   = frame.f_code

    method = code.co_name if code.co_name != '<module>' else 'main'
    ts     = datetime.now().strftime('%H:%M:%S')
    prefix = '%s[%s] [%s] %s %s line %i:' % (
        space, ts, level, code.co_filename.split('/')[-1], method, frame.f_lineno
    )

    final_msg = '\n%s\n%s%s' % (prefix, space, message)
    stream    = sys.stderr if level in ('ERROR', 'WARNING') else sys.stdout

    print(final_msg, file=stream)

    if level in ('WARNING', 'ERROR'):
        file_url = '%s#L%i' % (code.co_filename, frame.f_lineno)
        print(space, file_url, file=stream)
