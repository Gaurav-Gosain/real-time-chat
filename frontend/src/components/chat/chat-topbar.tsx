import React from "react";
import { ModeToggle } from "../ui/theme-toggle";

export default function ChatTopbar() {
  return (
    <div className="w-full h-20 flex p-4 justify-between items-center border-b">
      <div></div>
      <ModeToggle />
    </div>
  );
}
