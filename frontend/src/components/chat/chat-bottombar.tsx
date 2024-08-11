import { FileImage, Paperclip, SendHorizontal } from "lucide-react";
import Link from "next/link";
import React, { useRef, useState } from "react";
import { buttonVariants } from "../ui/button";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";
import { Message, loggedInUserData } from "@/data/data";
import { Textarea } from "../ui/textarea";

interface ChatBottombarProps {
  // sendMessage: (newMessage: Message) => void;
  isMobile: boolean;
  chatBoxText: string;
}

export const BottombarIcons = [{ icon: FileImage }, { icon: Paperclip }];

export default function ChatBottombar({
  // sendMessage,
  chatBoxText,
}: ChatBottombarProps) {
  const [message, setMessage] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(event.target.value);
  };

  // const handleSend = () => {
  //   // TODO: send the message to db
  //   if (message.trim()) {
  //     const newMessage: Message = {
  //       id: message.length + 1,
  //       name: loggedInUserData.name,
  //       avatar: loggedInUserData.avatar,
  //       message: message.trim(),
  //     };
  //     // sendMessage(newMessage);
  //     setMessage("");
  //
  //     if (inputRef.current) {
  //       inputRef.current.focus();
  //     }
  //   }
  // };

  return (
    <div className="p-2 flex justify-between w-full items-center gap-2">
      <AnimatePresence initial={false}>
        <motion.div
          key="input"
          className="w-full relative"
          layout
          initial={{ opacity: 0, scale: 1 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 1 }}
          transition={{
            opacity: { duration: 0.05 },
            layout: {
              type: "spring",
              bounce: 0.15,
            },
          }}
        >
          <Textarea
            autoComplete="off"
            value={chatBoxText}
            ref={inputRef}
            onKeyDown={() => {}}
            onChange={() => {}}
            name="message"
            placeholder="Aa"
            className="min-h-[auto] w-full border rounded-full flex items-center h-9 resize-none overflow-hidden bg-background"
          ></Textarea>
        </motion.div>

        {chatBoxText.trim() && (
          <Link
            href="#"
            className={cn(
              buttonVariants({ variant: "ghost", size: "icon" }),
              "h-9 w-9",
              "dark:bg-muted dark:text-muted-foreground dark:hover:bg-muted dark:hover:text-white shrink-0",
            )}
          >
            <SendHorizontal size={20} className="text-muted-foreground" />
          </Link>
        )}
      </AnimatePresence>
    </div>
  );
}
