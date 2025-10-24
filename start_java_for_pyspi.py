# start_java_for_pyspi.py
from __future__ import annotations
import pathlib, inspect
import jpype
import pyspi  # to locate the bundled JIDT jar

def ensure_java_started() -> None:
    if jpype.isJVMStarted():
        return
    # Adjust if your JDK path ever changes:
    jvm = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.8.9-hotspot\bin\server\jvm.dll"
    jar = str(pathlib.Path(inspect.getfile(pyspi)).parent / "lib" / "jidt" / "infodynamics.jar")
    jpype.startJVM(jvm, classpath=[jar])

# auto-start on import
ensure_java_started()
